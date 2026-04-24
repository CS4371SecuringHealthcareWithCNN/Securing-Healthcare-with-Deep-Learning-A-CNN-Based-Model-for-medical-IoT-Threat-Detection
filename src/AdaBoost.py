import os
import argparse
import numpy as np
import pandas as pd
import copy
import time
import joblib

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# ─── Attack category mappings ─────────────────────────────────────────────────
ATTACK_CATEGORIES_19 = {
    'ARP_Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT-DDoS-Connect_Flood',
    'MQTT-DDoS-Publish_Flood': 'MQTT-DDoS-Publish_Flood',
    'MQTT-DoS-Connect_Flood': 'MQTT-DoS-Connect_Flood',
    'MQTT-DoS-Publish_Flood': 'MQTT-DoS-Publish_Flood',
    'MQTT-Malformed_Data': 'MQTT-Malformed_Data',
    'Recon-OS_Scan': 'Recon-OS_Scan',
    'Recon-Ping_Sweep': 'Recon-Ping_Sweep',
    'Recon-Port_Scan': 'Recon-Port_Scan',
    'Recon-VulScan': 'Recon-VulScan',
    'TCP_IP-DDoS-ICMP': 'DDoS-ICMP',
    'TCP_IP-DDoS-SYN': 'DDoS-SYN',
    'TCP_IP-DDoS-TCP': 'DDoS-TCP',
    'TCP_IP-DDoS-UDP': 'DDoS-UDP',
    'TCP_IP-DoS-ICMP': 'DoS-ICMP',
    'TCP_IP-DoS-SYN': 'DoS-SYN',
    'TCP_IP-DoS-TCP': 'DoS-TCP',
    'TCP_IP-DoS-UDP': 'DoS-UDP',
    'Benign': 'Benign'
}

ATTACK_CATEGORIES_6 = {
    'Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT', 'MQTT-DDoS-Publish_Flood': 'MQTT',
    'MQTT-DoS-Connect_Flood': 'MQTT',  'MQTT-DoS-Publish_Flood': 'MQTT',
    'MQTT-Malformed_Data': 'MQTT',
    'Recon-OS_Scan': 'Recon', 'Recon-Ping_Sweep': 'Recon',
    'Recon-Port_Scan': 'Recon', 'Recon-VulScan': 'Recon',
    'DDoS-ICMP': 'DDoS', 'DDoS-SYN': 'DDoS', 'DDoS-TCP': 'DDoS', 'DDoS-UDP': 'DDoS',
    'DoS-ICMP': 'DoS',  'DoS-SYN': 'DoS',  'DoS-TCP': 'DoS',  'DoS-UDP': 'DoS',
    'Benign': 'Benign'
}

ATTACK_CATEGORIES_2 = {k: ('attack' if k != 'Benign' else 'Benign') for k in ATTACK_CATEGORIES_19}

def get_attack_category(file_name, class_config):
    categories = ATTACK_CATEGORIES_2 if class_config == 2 else (
        ATTACK_CATEGORIES_6 if class_config == 6 else ATTACK_CATEGORIES_19
    )
    for key in categories:
        if key in file_name:
            return categories[key]

def load_data(data_dir, class_config):
    train_files = [f"{data_dir}/train/{f}" for f in os.listdir(f"{data_dir}/train") if f.endswith('.csv')]
    test_files  = [f"{data_dir}/test/{f}"  for f in os.listdir(f"{data_dir}/test")  if f.endswith('.csv')]

    train_df = pd.concat([pd.read_csv(f).assign(file=f) for f in train_files], ignore_index=True)
    test_df  = pd.concat([pd.read_csv(f).assign(file=f) for f in test_files],  ignore_index=True)

    train_df['Attack_Type'] = train_df['file'].apply(lambda x: get_attack_category(x, class_config))
    test_df['Attack_Type']  = test_df['file'].apply(lambda x: get_attack_category(x, class_config))

    X_train = train_df.drop(['Attack_Type', 'file'], axis=1)
    y_train = train_df['Attack_Type']
    X_test  = test_df.drop(['Attack_Type', 'file'], axis=1)
    y_test  = test_df['Attack_Type']

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc  = label_encoder.transform(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train_enc, y_test_enc, label_encoder

def quantize_array(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.int8), float(mn), 0.0
    scale = (mx - mn) / 255.0
    zero_point = int(-mn / scale) - 128
    zero_point = max(-128, min(127, zero_point))
    q = np.clip(np.round(arr / scale) + zero_point, -128, 127).astype(np.int8)
    return q, scale, zero_point

def dequantize_array(q_arr, scale, zero_point):
    return (q_arr.astype(np.float32) - zero_point) * scale

def apply_drq(model):
    """INT8 Dynamic Range Quantization."""
    q_model = copy.deepcopy(model)
    q_weights, w_scale, w_zp = quantize_array(q_model.estimator_weights_)
    q_model.estimator_weights_ = dequantize_array(q_weights, w_scale, w_zp)
    for est in q_model.estimators_:
        tree = est.tree_
        q_t, t_scale, t_zp = quantize_array(tree.threshold)
        tree.threshold[:] = dequantize_array(q_t, t_scale, t_zp)
        q_v, v_scale, v_zp = quantize_array(tree.value)
        tree.value[:] = dequantize_array(q_v, v_scale, v_zp)
    return q_model, q_weights

def apply_fp16(model):
    """FP16 Half-Precision Quantization."""
    q_model = copy.deepcopy(model)
    q_model.estimator_weights_ = q_model.estimator_weights_.astype(np.float16)
    for est in q_model.estimators_:
        tree = est.tree_
        tree.threshold[:] = tree.threshold.astype(np.float16)
        tree.value[:] = tree.value.astype(np.float16)
    return q_model

def evaluate(model, X_test, y_test, label_encoder, label="Model"):
    t0 = time.time()
    y_pred_enc = model.predict(X_test)
    elapsed = time.time() - t0

    y_pred = label_encoder.inverse_transform(y_pred_enc)
    y_true = label_encoder.inverse_transform(y_test)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Precision:      {prec:.4f}")
    print(f"  Recall:         {rec:.4f}")
    print(f"  F1-Score:       {f1:.4f}")
    print(f"  Inference Time: {elapsed*1000:.2f} ms")
    print(f"\nClassification Report:\n{classification_report(y_true, y_pred, zero_division=0)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

    return acc, prec, rec, f1, elapsed

def model_size(model, q_weights=None):
    size = sum(e.tree_.threshold.nbytes + e.tree_.value.nbytes for e in model.estimators_)
    size += q_weights.nbytes if q_weights is not None else model.estimator_weights_.nbytes
    return size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaBoost with FP16 and INT8 DRQ on CICIoMT2024")
    parser.add_argument("--class_config", type=int, choices=[2, 6, 19], default=2)
    parser.add_argument("--n_estimators", type=int, default=50)
    args = parser.parse_args()

    data_dir = r"C:\Users\suhja\OneDrive\Documents\GitHub\Securing-Healthcare-with-Deep-Learning-A-CNN-Based-Model-for-medical-IoT-Threat-Detection\data"
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"adaboost_model_class{args.class_config}.pkl")

    print(f"Loading data [{args.class_config}-class config]...")
    X_train, X_test, y_train, y_test, label_encoder = load_data(data_dir, args.class_config)

    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}...")
        model = joblib.load(model_path)
    else:
        print(f"\nTraining AdaBoost (n_estimators={args.n_estimators})...")
        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=args.n_estimators,
            learning_rate=1.0,
            random_state=42
        )
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

    # FP32 Baseline
    size_fp32 = model_size(model)
    acc_fp32, prec_fp32, rec_fp32, f1_fp32, t_fp32 = evaluate(
        model, X_test, y_test, label_encoder, label="FP32 (Original AdaBoost)"
    )
    print(f"  Model Size: {size_fp32} bytes ({size_fp32/1024:.2f} KB)")

    # FP16
    print("\nApplying FP16 (Half Precision) Quantization...")
    fp16_model = apply_fp16(model)
    size_fp16 = model_size(fp16_model)
    acc_fp16, prec_fp16, rec_fp16, f1_fp16, t_fp16 = evaluate(
        fp16_model, X_test, y_test, label_encoder, label="FP16 (Half Precision)"
    )
    print(f"  Model Size: {size_fp16} bytes ({size_fp16/1024:.2f} KB)")

    # INT8 DRQ
    print("\nApplying INT8 Dynamic Range Quantization...")
    int8_model, q_weights = apply_drq(model)
    size_int8 = model_size(int8_model, q_weights)
    acc_int8, prec_int8, rec_int8, f1_int8, t_int8 = evaluate(
        int8_model, X_test, y_test, label_encoder, label="INT8 (Dynamic Range Quantization)"
    )
    print(f"  Model Size: {size_int8} bytes ({size_int8/1024:.2f} KB)")

    # Summary Table
    print(f"\n{'='*40}")
    print("  Quantization Impact Summary")
    print(f"{'='*40}")
    print(f"  {'Metric':<25} {'FP32':>8} {'FP16':>8} {'INT8':>8}")
    print(f"  {'-'*49}")
    print(f"  {'Accuracy':<25} {acc_fp32:>8.4f} {acc_fp16:>8.4f} {acc_int8:>8.4f}")
    print(f"  {'Precision':<25} {prec_fp32:>8.4f} {prec_fp16:>8.4f} {prec_int8:>8.4f}")
    print(f"  {'Recall':<25} {rec_fp32:>8.4f} {rec_fp16:>8.4f} {rec_int8:>8.4f}")
    print(f"  {'F1-Score':<25} {f1_fp32:>8.4f} {f1_fp16:>8.4f} {f1_int8:>8.4f}")
    print(f"  {'Model Size (bytes)':<25} {size_fp32:>8} {size_fp16:>8} {size_int8:>8}")
    print(f"  {'Inference Time (ms)':<25} {t_fp32*1000:>8.1f} {t_fp16*1000:>8.1f} {t_int8*1000:>8.1f}")
    print(f"  {'-'*49}")
    print(f"  {'FP16 Accuracy Drop':<25} {(acc_fp32-acc_fp16)*100:>7.2f}%")
    print(f"  {'INT8 Accuracy Drop':<25} {(acc_fp32-acc_int8)*100:>7.2f}%")
    print(f"  {'FP16 Size Reduction':<25} {(1-size_fp16/size_fp32)*100:>7.1f}%")
    print(f"  {'INT8 Size Reduction':<25} {(1-size_int8/size_fp32)*100:>7.1f}%")