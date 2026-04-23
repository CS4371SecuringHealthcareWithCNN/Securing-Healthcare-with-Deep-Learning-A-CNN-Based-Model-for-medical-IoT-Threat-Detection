import argparse
import os
from typing import Dict, List
 
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# TFLite conversion functions
# Each function takes the Keras model and returns the compressed model as a raw bytes blob.

def _representative_dataset_factory(X_calib: np.ndarray):
    samples = X_calib.astype(np.float32)
    def gen():
        for s in samples:
            yield [s[np.newaxis, ...]]
    return gen

# Converts to TFlite without quantization. Proves conversion is lossless
def convert_fp32(keras_model) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    return conv.convert()

# Converts weights to 8-bit integers, but activations remain fp32
def convert_dynamic_range(keras_model) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    return conv.convert()

# Converts weights to 16-bit floats, but activations remain fp32
def convert_float16(keras_model) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    return conv.convert()

# Converts weights and activations to 8-bit integers. Uses a representative dataset for calibration due to changes in activation ranges.
def convert_int8(keras_model, X_calib: np.ndarray, int_io: bool = False) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = _representative_dataset_factory(X_calib)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if int_io:
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()

# Prediction and metric functions

def predict_keras(keras_model, X_test: np.ndarray) -> np.ndarray:
    probs = keras_model.predict(X_test, batch_size=256, verbose=0)
    return probs.argmax(axis=1)

def predict_tflite(tflite_bytes: bytes, X_test: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    
    # Get input and output details for quantization handling
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_dtype = inp['dtype'] # np.float32, np.int8, or np.uint8
    in_scale, in_zero = inp.get("quantization", (0.0, 0)) # int8 scale factor
    out_dtype = out['dtype']
    out_scale, out_zero = out.get("quantization", (0.0, 0))    
    
    preds = np.empty(len(X_test), dtype=np.int64)
    for i, sample in enumerate(X_test):
        # Reshape sample to match interpreter input
        x = sample[np.newaxis, ...].astype(np.float32)
        # Quantize input for int8 models, otherwise pass as float32
        if in_dtype in (np.int8, np.uint8) and in_scale > 0:
            x_q = x / in_scale + in_zero
            x_q = np.clip(np.round(x_q), np.iinfo(in_dtype).min, np.iinfo(in_dtype).max)
            interpreter.set_tensor(inp['index'], x_q.astype(in_dtype))
        else:
            interpreter.set_tensor(inp['index'], x.astype(in_dtype))
        interpreter.invoke()
        o = interpreter.get_tensor(out['index'])
        # Dequantize output for int8 models so argmax is meaningful, otherwise use as is
        if out_dtype in (np.int8, np.uint8) and out_scale > 0:
            o = (o.astype(np.float32) - out_zero) * out_scale
        preds[i] = int(np.argmax(o[0]))
    return preds
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict:
    labels = np.arange(num_classes)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class_f1": f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels),
        "per_class_precision": precision_score(y_true, y_pred, average=None, zero_division=0, labels=labels),
        "per_class_recall": recall_score(y_true, y_pred, average=None, zero_division=0, labels=labels),
    }
    
def load_class_names(path: str, num_classes: int) -> List[str]:
    if path is None:
        return [f"class_{i:02d}" for i in range(num_classes)]
    if path.endswith(".npy"):
        names = np.load(path, allow_pickle=True)
    else:  # plain-text, one class per line
        with open(path) as f:
            names = [line.strip() for line in f if line.strip()]
    names = [str(n) for n in names]
    # Sanity check to ensure the number of names match the model's output dim
    if len(names) != num_classes:
        raise ValueError(
            f"class_names has {len(names)} entries but model outputs {num_classes} classes."
        )
    return names

# Main function

def main():
    # Argument parsing 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--X_test", required=True)
    parser.add_argument("--y_test", required=True)
    parser.add_argument("--X_calib", required=True)
    parser.add_argument("--class_names", default=None, help="Optional path to class names file (npy or txt)")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--subsample_test", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and data
    print("\n  Loading model and data...")
    baseline = tf.keras.models.load_model(args.model)
    baseline_size = os.path.getsize(args.model)
    baseline.summary(print_fn=lambda s: print("  " + s))
    
    X_test = np.load(args.X_test).astype(np.float32)
    y_test = np.load(args.y_test)
    X_calib = np.load(args.X_calib).astype(np.float32)
    
    if y_test.ndim == 2:
        y_test_int = y_test.argmax(axis=1)
    else:
        y_test_int = y_test.astype(np.int64)
        
    # Figure out class count from models output shape then load names
    num_classes = baseline.output_shape[-1]
    class_names = load_class_names(args.class_names, num_classes)
    print(f"  X_test: {X_test.shape} y_test: {y_test.shape}")
    print(f"  {num_classes} classes (first 5): {', '.join(class_names[:5])} ")
    
    # Optional subsampling for faster evaluation
    if args.subsample_test and args.subsample_test < len(X_test):
        idx = np.random.RandomState(42).choice(len(X_test), args.subsample_test, replace=False)
        X_test = X_test[idx]
        y_test_int = y_test_int[idx]
        print(f"  Subsampled test set to {len(X_test)}.")
        
    # Run TFLite conversions
    # We write each one to disk so final files can be deliverable artifacts
    print("\n  Converting to TFLite formats for quantization...")
    conversions = {
        "baseline_fp32_tflite": convert_fp32(baseline),
        "dynamic_range": convert_dynamic_range(baseline),
        "float16": convert_float16(baseline),
        "full_int8": convert_int8(baseline, X_calib, int_io=False),
        "full_int8_io": convert_int8(baseline, X_calib, int_io=True),
    }
    for name, blob in conversions.items():
        path = os.path.join(args.output_dir, f"{name}.tflite")
        with open(path, "wb") as f:
            f.write(blob)
        print(f"  {name:<24} {len(blob) / 1024:>8.1f} KB -> {path}")
        
    # Evaluate each model on the test set and get metrics
    print("  Evaluating models on test set...")
    results: Dict[str, Dict] = {}
    
    # Baseline Keras model
    print("  Baseline Keras")
    preds = predict_keras(baseline, X_test)
    m = compute_metrics(y_test_int, preds, num_classes)
    m["size_kb"] = baseline_size / 1024.0
    results["baseline_keras"] = m
    
    for name, blob in conversions.items():
        print(f"   - {name}")
        preds = predict_tflite(blob, X_test)
        m = compute_metrics(y_test_int, preds, num_classes)
        m["size_kb"] = len(blob) / 1024.0
        results[name] = m
    
    # Count test samples per class for per-class table
    support = np.bincount(y_test_int, minlength=num_classes)
    
    # Save and print results
    print("\n" + "=" * 100)
    print("SUMMARY — overall metrics per variant")
    print("=" * 100)
    header = (
        f"{'Model':<22} {'Size(KB)':>10} {'Ratio':>7} {'Acc':>7} "
        f"{'F1-w':>7} {'F1-macro':>10} {'Prec-w':>8} {'Prec-m':>8} "
        f"{'Rec-w':>7} {'Rec-m':>7}"
    )
    print(header)
    print("-" * len(header))
 
    # Compression ratio is always relative to the Keras baseline.
    base_size = results["baseline_keras"]["size_kb"]
    for name, m in results.items():
        ratio = base_size / m["size_kb"] if m["size_kb"] > 0 else float("inf")
        print(
            f"{name:<22} {m['size_kb']:>10.2f} {ratio:>6.2f}x "
            f"{m['accuracy']:>7.4f} {m['f1_weighted']:>7.4f} "
            f"{m['f1_macro']:>10.4f} {m['precision_weighted']:>8.4f} "
            f"{m['precision_macro']:>8.4f} {m['recall_weighted']:>7.4f} "
            f"{m['recall_macro']:>7.4f}"
        )
    # Per-class F1 table
    print("\n" + "=" * 110)
    print("PER-CLASS F1 - sorted by support ascending")
    print("=" * 110)
    variant_names = list(results.keys())
    # Shorten variant names for simpler display
    short_variants = [v.replace("baseline_", "").replace("_tflite", "")[:12] for v in variant_names]
    widest = max(len(c) for c in class_names)
    pc_header = (f"{'Class':<{widest+2}} {'Support':>9}  "+ "  ".join(f"{v:>12}" for v in short_variants))
    print(pc_header)
    print("-" * len(pc_header))
 
    order = np.argsort(support)  # ascending
    for i in order:
        row = (f"{class_names[i]:<{widest+2}} {int(support[i]):>9}  " + "  ".join(f"{results[v]['per_class_f1'][i]:>12.4f}"for v in variant_names))
        print(row)
        
    # Summary CSV
    summary_path = os.path.join(args.output_dir, "results.csv")
    with open(summary_path, "w") as f:
        f.write(
            "model,size_kb,compression_ratio,accuracy,"
            "precision_weighted,recall_weighted,f1_weighted,"
            "precision_macro,recall_macro,f1_macro\n"
        )
        for name, m in results.items():
            ratio = base_size / m["size_kb"] if m["size_kb"] > 0 else 0
            f.write(
                f"{name},{m['size_kb']:.4f},{ratio:.4f},"
                f"{m['accuracy']:.6f},"
                f"{m['precision_weighted']:.6f},{m['recall_weighted']:.6f},{m['f1_weighted']:.6f},"
                f"{m['precision_macro']:.6f},{m['recall_macro']:.6f},{m['f1_macro']:.6f}\n"
            )
    
    # Per-class CSVs
    # Quickly plots if compression hurt class x more than y
    def _write_per_class(metric_key: str, fname: str):
        path = os.path.join(args.output_dir, fname)
        with open(path, "w") as f:
            # Header: class, support, then one column per variant.
            f.write("class,support," + ",".join(variant_names) + "\n")
            for i in range(num_classes):
                row = (f"{class_names[i]},{int(support[i])}," + ",".join(f"{results[v][metric_key][i]:.6f}" for v in variant_names))
                f.write(row + "\n")
        return path
    
    f1_csv = _write_per_class("per_class_f1", "per_class_f1.csv")
    prec_csv = _write_per_class("per_class_precision", "per_class_precision.csv")
    rec_csv = _write_per_class("per_class_recall", "per_class_recall.csv")
 
    print(f"\nSaved: {summary_path}")
    print(f"       {f1_csv}")
    print(f"       {prec_csv}")
    print(f"       {rec_csv}")
    
if __name__ == "__main__":
    main()