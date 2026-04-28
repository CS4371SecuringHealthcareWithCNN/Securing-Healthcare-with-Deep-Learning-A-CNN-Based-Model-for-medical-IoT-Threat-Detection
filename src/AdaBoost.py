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
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)

# ─── Attack category mappings ─────────────────────────────────────────────────
ATTACK_CATEGORIES_19 = {
    'ARP_Spoofing':'Spoofing','MQTT-DDoS-Connect_Flood':'MQTT-DDoS-Connect_Flood',
    'MQTT-DDoS-Publish_Flood':'MQTT-DDoS-Publish_Flood','MQTT-DoS-Connect_Flood':'MQTT-DoS-Connect_Flood',
    'MQTT-DoS-Publish_Flood':'MQTT-DoS-Publish_Flood','MQTT-Malformed_Data':'MQTT-Malformed_Data',
    'Recon-OS_Scan':'Recon-OS_Scan','Recon-Ping_Sweep':'Recon-Ping_Sweep',
    'Recon-Port_Scan':'Recon-Port_Scan','Recon-VulScan':'Recon-VulScan',
    'TCP_IP-DDoS-ICMP':'DDoS-ICMP','TCP_IP-DDoS-SYN':'DDoS-SYN','TCP_IP-DDoS-TCP':'DDoS-TCP','TCP_IP-DDoS-UDP':'DDoS-UDP',
    'TCP_IP-DoS-ICMP':'DoS-ICMP','TCP_IP-DoS-SYN':'DoS-SYN','TCP_IP-DoS-TCP':'DoS-TCP','TCP_IP-DoS-UDP':'DoS-UDP',
    'Benign':'Benign'
}
ATTACK_CATEGORIES_6 = {
    'Spoofing':'Spoofing','MQTT-DDoS-Connect_Flood':'MQTT','MQTT-DDoS-Publish_Flood':'MQTT',
    'MQTT-DoS-Connect_Flood':'MQTT','MQTT-DoS-Publish_Flood':'MQTT','MQTT-Malformed_Data':'MQTT',
    'Recon-OS_Scan':'Recon','Recon-Ping_Sweep':'Recon','Recon-Port_Scan':'Recon','Recon-VulScan':'Recon',
    'DDoS-ICMP':'DDoS','DDoS-SYN':'DDoS','DDoS-TCP':'DDoS','DDoS-UDP':'DDoS',
    'DoS-ICMP':'DoS','DoS-SYN':'DoS','DoS-TCP':'DoS','DoS-UDP':'DoS','Benign':'Benign'
}
ATTACK_CATEGORIES_2 = {k: ('attack' if k != 'Benign' else 'Benign') for k in ATTACK_CATEGORIES_19}

# picks the right lookup table based on class config argument
# then searches the filename for a matching key and returns the label
def get_attack_category(file_name, class_config):
    cats = ATTACK_CATEGORIES_2 if class_config == 2 else (
           ATTACK_CATEGORIES_6 if class_config == 6 else ATTACK_CATEGORIES_19)
    return next((v for k, v in cats.items() if k in file_name), None)

# Finds all CSV files in the data/train and data/test
# Reads and concatenates all 19 train and test CSV's into one big dataframe
def load_data(data_dir, class_config):
    def read_dir(split):
        files = [f"{data_dir}/{split}/{f}" for f in os.listdir(f"{data_dir}/{split}") if f.endswith('.csv')]
        df = pd.concat([pd.read_csv(f).assign(file=f) for f in files], ignore_index=True)
        df['Attack_Type'] = df['file'].apply(lambda x: get_attack_category(x, class_config))
        return df.drop(['Attack_Type','file'], axis=1), df['Attack_Type']
    X_train, y_train = read_dir('train')
    X_test,  y_test  = read_dir('test')
    label_encoder = LabelEncoder()  # assigns the correct label to each row based on which file it came from
    y_train, y_test = label_encoder.fit_transform(y_train), label_encoder.transform(y_test)
    scaler = StandardScaler()  # normalize all features to mean=0 and std=1
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test, label_encoder

# Compresses a float32 array into 8-bit integers
def quantize_array(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:  # maps the full range of values max and min into -128 and 127
        return np.zeros_like(arr, dtype=np.int8), float(mn), 0.0  # quantized array + scale and zero_point for future reversal
    scale = (mx - mn) / 255.0
    zp = max(-128, min(127, int(-mn / scale) - 128))
    return np.clip(np.round(arr / scale) + zp, -128, 127).astype(np.int8), scale, zp

# Reverses quantize_array
def dequantize_array(q, scale, zp):
    return (q.astype(np.float32) - zp) * scale

# FP16 Half-Precision Quantization
def apply_fp16(model):
    m = copy.deepcopy(model)  # deep copy so original FP32 model is untouched
    m.estimator_weights_ = m.estimator_weights_.astype(np.float16)
    for est in m.estimators_:  # converts all weights and tree values from 32-bit to 16-bit
        est.tree_.threshold[:] = est.tree_.threshold.astype(np.float16)
        est.tree_.value[:]     = est.tree_.value.astype(np.float16)
    return m

# INT8 Dynamic Range Quantization
def apply_drq(model):
    m = copy.deepcopy(model)  # deep copy so original FP32 model is untouched
    qw, ws, wz = quantize_array(m.estimator_weights_)  # quantize estimator weights
    m.estimator_weights_ = dequantize_array(qw, ws, wz)
    for est in m.estimators_:  # quantizes split thresholds for each decision tree
        t = est.tree_
        qt, ts, tz = quantize_array(t.threshold); t.threshold[:] = dequantize_array(qt, ts, tz)
        qv, vs, vz = quantize_array(t.value);     t.value[:]     = dequantize_array(qv, vs, vz)
    return m, qw

# calculates memory footprint of the model in bytes
def model_size(model, qw=None):
    size = sum(e.tree_.threshold.nbytes + e.tree_.value.nbytes for e in model.estimators_)  # sum threshold + value arrays across all trees
    return size + (qw.nbytes if qw is not None else model.estimator_weights_.nbytes)  # adds estimator weights array

# converts integer predictions back to string labels for readability, returns metrics silently
def evaluate(model, X_test, y_test, le):
    t0 = time.time()
    y_pred = le.inverse_transform(model.predict(X_test))
    elapsed = time.time() - t0
    y_true = le.inverse_transform(y_test)
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average='weighted', zero_division=0),  # of all attack predictions, how many were real
        recall_score(y_true, y_pred, average='weighted', zero_division=0),     # of all real attacks, how many were caught
        f1_score(y_true, y_pred, average='weighted', zero_division=0),
        elapsed,
        classification_report(y_true, y_pred, zero_division=0)
    )

# formats raw seconds into a more readable format
def fmt_time(s):
    return f"{s:.1f}s" if s < 60 else f"{int(s//60)}m {s%60:.1f}s"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaBoost with FP16 and INT8 DRQ on CICIoMT2024")
    parser.add_argument("--class_config", type=int, choices=[2, 6, 19], default=2)
    parser.add_argument("--n_estimators", type=int, default=50)
    args = parser.parse_args()

    data_dir   = r"C:\Users\suhja\OneDrive\Documents\GitHub\Securing-Healthcare-with-Deep-Learning-A-CNN-Based-Model-for-medical-IoT-Threat-Detection\data"
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"adaboost_model_class{args.class_config}.pkl")

    print(f"Loading data [{args.class_config}-class config]...")
    X_train, X_test, y_train, y_test, le = load_data(data_dir, args.class_config)

    t_train = 0.0
    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}...")
        model = joblib.load(model_path)
    else:
        print(f"Training AdaBoost (n_estimators={args.n_estimators})...")
        t0 = time.time()
        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=args.n_estimators, learning_rate=0.5, random_state=42
        )
        model.fit(X_train, y_train)
        t_train = time.time() - t0
        joblib.dump(model, model_path)
        print(f"Trained in {fmt_time(t_train)} — saved to {model_path}")

    # FP32 Baseline
    print("Evaluating FP32...")
    size_fp32 = model_size(model)
    acc_fp32, prec_fp32, rec_fp32, f1_fp32, t_fp32, rep_fp32 = evaluate(model, X_test, y_test, le)

    # FP16
    print("Applying FP16 (Half Precision) Quantization...")
    fp16_model = apply_fp16(model)
    size_fp16 = model_size(fp16_model)
    acc_fp16, prec_fp16, rec_fp16, f1_fp16, t_fp16, rep_fp16 = evaluate(fp16_model, X_test, y_test, le)

    # INT8 DRQ
    print("Applying INT8 Dynamic Range Quantization...")
    int8_model, qw = apply_drq(model)
    size_int8 = model_size(int8_model, qw)
    acc_int8, prec_int8, rec_int8, f1_int8, t_int8, rep_int8 = evaluate(int8_model, X_test, y_test, le)

    # Before and After Quantization results
    print(f"\n{'='*60}")
    print("  BEFORE QUANTIZATION — FP32 (Original AdaBoost)")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc_fp32:.4f}  Precision: {prec_fp32:.4f}  Recall: {rec_fp32:.4f}  F1: {f1_fp32:.4f}")
    print(f"  Model Size: {size_fp32/1024:.2f} KB  |  Inference: {t_fp32*1000:.2f} ms")
    print(f"\nClassification Report:\n{rep_fp32}")

    print(f"\n{'='*60}")
    print("  AFTER QUANTIZATION — FP16 (Half Precision)")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc_fp16:.4f} ({(acc_fp16-acc_fp32)*100:+.4f}%)  F1: {f1_fp16:.4f} ({(f1_fp16-f1_fp32)*100:+.4f}%)")
    print(f"  Model Size: {size_fp16/1024:.2f} KB ({(1-size_fp16/size_fp32)*100:.1f}% smaller, {size_fp32/size_fp16:.2f}x)  |  Inference: {t_fp16*1000:.2f} ms")
    print(f"\nClassification Report:\n{rep_fp16}")

    print(f"\n{'='*60}")
    print("  AFTER QUANTIZATION — INT8 (Dynamic Range Quantization)")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc_int8:.4f} ({(acc_int8-acc_fp32)*100:+.4f}%)  F1: {f1_int8:.4f} ({(f1_int8-f1_fp32)*100:+.4f}%)")
    print(f"  Model Size: {size_int8/1024:.2f} KB ({(1-size_int8/size_fp32)*100:.1f}% smaller, {size_fp32/size_int8:.2f}x)  |  Inference: {t_int8*1000:.2f} ms")
    print(f"\nClassification Report:\n{rep_int8}")

    # Quantization Summary
    print(f"\n{'='*40}\n  Quantization Impact Summary\n{'='*40}")
    print(f"  {'Metric':<25} {'FP32':>8} {'FP16':>8} {'INT8':>8}")
    print(f"  {'-'*49}")
    for name, vals in [('Accuracy',(acc_fp32,acc_fp16,acc_int8)), ('Precision',(prec_fp32,prec_fp16,prec_int8)),
                       ('Recall',(rec_fp32,rec_fp16,rec_int8)),   ('F1-Score',(f1_fp32,f1_fp16,f1_int8))]:
        print(f"  {name:<25} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f}")
    print(f"  {'Model Size (KB)':<25} {size_fp32/1024:>8.2f} {size_fp16/1024:>8.2f} {size_int8/1024:>8.2f}")
    print(f"  {'Inference Time (ms)':<25} {t_fp32*1000:>8.1f} {t_fp16*1000:>8.1f} {t_int8*1000:>8.1f}")
    print(f"  {'-'*49}")
    print(f"  {'FP16 Accuracy Drop':<25} {(acc_fp32-acc_fp16)*100:>7.2f}%")
    print(f"  {'INT8 Accuracy Drop':<25} {(acc_fp32-acc_int8)*100:>7.2f}%")
    print(f"  {'FP16 Size Reduction':<25} {(1-size_fp16/size_fp32)*100:>7.1f}%")
    print(f"  {'INT8 Size Reduction':<25} {(1-size_int8/size_fp32)*100:>7.1f}%")