import argparse
import os
import numpy as np
import tensorflow as tf
from data_loader import get_attack_category, ATTACK_CATEGORIES_2, ATTACK_CATEGORIES_6, ATTACK_CATEGORIES_19
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pandas as pd
import psutil
import time


def load_data_for_rf(data_dir, class_config):
    """Load and preprocess data for Random Forest (no reshaping or one-hot encoding)."""
    train_files = [f"{data_dir}/train/{f}" for f in os.listdir(f"{data_dir}/train") if f.endswith('.csv')]
    test_files = [f"{data_dir}/test/{f}" for f in os.listdir(f"{data_dir}/test") if f.endswith('.csv')]

    train_df = pd.concat([pd.read_csv(f).assign(file=f) for f in train_files], ignore_index=True)
    test_df = pd.concat([pd.read_csv(f).assign(file=f) for f in test_files], ignore_index=True)

    train_df['Attack_Type'] = train_df['file'].apply(lambda x: get_attack_category(x, class_config))
    test_df['Attack_Type'] = test_df['file'].apply(lambda x: get_attack_category(x, class_config))

    X_train = train_df.drop(['Attack_Type', 'file'], axis=1)
    y_train = train_df['Attack_Type']
    X_test = test_df.drop(['Attack_Type', 'file'], axis=1)
    y_test = test_df['Attack_Type']

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train_encoded, y_test_encoded, label_encoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a Random Forest for network intrusion detection.")
    parser.add_argument("--class_config", type=int, choices=[2, 6, 19], default=2,
                        help="Number of classes for classification (2, 6, or 19)")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees in the Random Forest")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')

    print("Loading data...")
    X_train, X_test, y_train, y_test, label_encoder = load_data_for_rf(data_dir, args.class_config)

    print(f"Training Random Forest with {args.n_estimators} trees...")
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    y_test_decoded = label_encoder.inverse_transform(y_test)

    accuracy = accuracy_score(y_test_decoded, y_pred)
    precision = precision_score(y_test_decoded, y_pred, average='weighted')
    recall = recall_score(y_test_decoded, y_pred, average='weighted')
    f1 = f1_score(y_test_decoded, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("\nClassification Report:\n", classification_report(y_test_decoded, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test_decoded, y_pred))

#TensorFlow Lite Compression 
    print("\nConverting to TensorFlow Lite...")
    rf_probs = model.predict_proba(X_test)

    input_dim = rf_probs.shape[1]
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(input_dim, activation='softmax')
    ])

    tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    tf_model.fit(rf_probs, y_test, epochs=5, batch_size=32, verbose=0)

    tf_model.save('rf_model.keras')
    original_size = os.path.getsize('rf_model.keras')
    print(f"Original model size: {original_size / 1024:.2f} KB")

    # Dynamic Range Quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_drq = converter.convert()
    with open('rf_model_drq.tflite', 'wb') as f:
        f.write(tflite_drq)
    drq_size = os.path.getsize('rf_model_drq.tflite')
    print(f"\n--- Dynamic Range Quantization ---")
    print(f"Compressed model size: {drq_size / 1024:.2f} KB")
    print(f"Size reduction: {(1 - drq_size / original_size) * 100:.2f}%")

    # Float16 Quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter.convert()
    with open('rf_model_fp16.tflite', 'wb') as f:
        f.write(tflite_fp16)
    fp16_size = os.path.getsize('rf_model_fp16.tflite')
    print(f"\n--- Float16 Quantization ---")
    print(f"Compressed model size: {fp16_size / 1024:.2f} KB")
    print(f"Size reduction: {(1 - fp16_size / original_size) * 100:.2f}%")

    def run_tflite(model_path, rf_probs_input):
        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        inp = interp.get_input_details()
        out = interp.get_output_details()
        results = []
        for i in range(len(rf_probs_input)):
            input_data = np.array([rf_probs_input[i]], dtype=np.float32)
            interp.set_tensor(inp[0]['index'], input_data)
            interp.invoke()
            output = interp.get_tensor(out[0]['index'])
            results.append(np.argmax(output))
        return np.array(results)

    drq_preds = label_encoder.inverse_transform(run_tflite('rf_model_drq.tflite', rf_probs))
    fp16_preds = label_encoder.inverse_transform(run_tflite('rf_model_fp16.tflite', rf_probs))

    print(f"\nDRQ Accuracy: {accuracy_score(y_test_decoded, drq_preds):.4f}")
    print(f"Float16 Accuracy: {accuracy_score(y_test_decoded, fp16_preds):.4f}")

    # - Benchmarking -
    print("\nBenchmarking...")

    def benchmark_model(predict_fn, X, y_test_decoded, label_encoder, model_path):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        cpu_before = psutil.cpu_percent(interval=None)
        start_time = time.time()
        predictions = predict_fn(X)
        elapsed_time = time.time() - start_time
        cpu_after = psutil.cpu_percent(interval=0.1)
        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_added = mem_after - mem_before
        size_kb = os.path.getsize(model_path) / 1024 if os.path.exists(model_path) else 0
        preds_decoded = label_encoder.inverse_transform(predictions)
        acc = accuracy_score(y_test_decoded, preds_decoded)
        return {
            'Time (s)': round(elapsed_time, 3),
            'CPU%': round(cpu_after, 1),
            'RAM (MB)': round(mem_after, 1),
            'RAM added at runtime (MB)': round(mem_added, 3),
            'Size (KB)': round(size_kb, 1),
            'Accuracy': round(acc, 4)
        }

    def rf_predict(X):
        return model.predict(X)

    def drq_predict(X):
        return run_tflite('rf_model_drq.tflite', model.predict_proba(X))

    def fp16_predict(X):
        return run_tflite('rf_model_fp16.tflite', model.predict_proba(X))

    rf_results = benchmark_model(rf_predict, X_test, y_test_decoded, label_encoder, 'rf_model.keras')
    drq_results = benchmark_model(drq_predict, X_test, y_test_decoded, label_encoder, 'rf_model_drq.tflite')
    fp16_results = benchmark_model(fp16_predict, X_test, y_test_decoded, label_encoder, 'rf_model_fp16.tflite')

    print(f"\n--- Benchmark Summary ---")
    print(f"{'Metric':<30} {'Original RF':<20} {'DRQ':<20} {'Float16':<20}")
    print(f"{'Time (s)':<30} {rf_results['Time (s)']:<20} {drq_results['Time (s)']:<20} {fp16_results['Time (s)']:<20}")
    print(f"{'CPU%':<30} {rf_results['CPU%']:<20} {drq_results['CPU%']:<20} {fp16_results['CPU%']:<20}")
    print(f"{'RAM (MB)':<30} {rf_results['RAM (MB)']:<20} {drq_results['RAM (MB)']:<20} {fp16_results['RAM (MB)']:<20}")
    print(f"{'RAM added at runtime (MB)':<30} {rf_results['RAM added at runtime (MB)']:<20} {drq_results['RAM added at runtime (MB)']:<20} {fp16_results['RAM added at runtime (MB)']:<20}")
    print(f"{'Size (KB)':<30} {rf_results['Size (KB)']:<20} {drq_results['Size (KB)']:<20} {fp16_results['Size (KB)']:<20}")
    print(f"{'Accuracy':<30} {rf_results['Accuracy']:<20} {drq_results['Accuracy']:<20} {fp16_results['Accuracy']:<20}")