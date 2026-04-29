import argparse, os, time
import numpy as np
import psutil
import tensorflow as tf

def predict_keras(model, X):
    model.predict(X, batch_size=256, verbose=0)
    
def predict_tflite(path, X):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input = interpreter.get_input_details()[0]
    in_dtype, (in_scale, in_zero) = input["dtype"], input.get("quantization", (0.0, 0))
    for s in X:
        x = s[np.newaxis, ...].astype(np.float32)
        if in_dtype in (np.int8, np.uint8) and in_scale > 0:
            x = np.clip(np.round(x / in_scale + in_zero), np.iinfo(in_dtype).min, np.iinfo(in_dtype).max).astype(in_dtype)
        interpreter.set_tensor(input["index"], x.astype(in_dtype, copy=False))
        interpreter.invoke()

def util_summary(name, path, run_fn, X):
    p = psutil.Process()
    p.cpu_percent(None)
    rss0 = p.memory_info().rss
    t0 = time.perf_counter()
    run_fn(X)
    t = time.perf_counter() - t0
    return{
        "name": name,
        "seconds": t,
        "latency_ms": t/len(X)*100,
        "cpu_pct": p.cpu_percent(None),
        "rss_mb": p.memory_info().rss / 1024**2,
        "rss_delta_mb": (p.memory_info().rss - rss0) / 1024**2,
        "size_kb": os.path.getsize(path) / 1024
    }
    
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--X_test", required=True)
parser.add_argument("--output_dir", default="results")
parser.add_argument("--subsample_test", type=int, default=0)
args = parser.parse_args()

baseline = tf.keras.models.load_model(args.model)
X = np.load(args.X_test).astype(np.float32)
if args.subsample_test:
    n = min(args.subsample_test, len(X))
    X = X[np.random.RandomState(42).choice(len(X), n, replace=False)]
    
results = [util_summary("baseline_keras", args.model, lambda x: predict_keras(baseline, x), X)]
for name in ["baseline_fp32_tflite", "dynamic_range", "float16", "full_int8", "full_int8_io"]:
    path = os.path.join(args.output_dir, f"{name}.tflite")
    if os.path.exists(path):
        results.append(util_summary(name, path, lambda x, p=path: predict_tflite(p, x), X))
        
for r in results:
    print(r)
    
csv = os.path.join(args.output_dir, "resource_usage.csv")
with open(csv, "w") as f:
    f.write(",".join(results[0].keys()) + "\n")
    for r in results:
        f.write(",".join(str(v) for v in r.values()) + "\n")
print(f"Saved: {csv}")