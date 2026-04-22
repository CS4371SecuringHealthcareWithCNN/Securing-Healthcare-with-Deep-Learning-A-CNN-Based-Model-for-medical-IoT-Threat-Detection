import os
import argparse
import numpy as np
import pandas as pd
import copy
import time

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Attack category mappings (mirrors data loader)
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

ATTACK_CATEGORIES_2 = {k: {'attack' if k != 'Benign' else 'Benign'} for k in ATTACK_CATEGORIES_19}

def get_attack_category(file_name, class_config):
    categories = ATTACK_CATEGORIES_2 if class_config == 2 else (
        ATTACK_CATEGORIES_6 if class_config == 6 else ATTACK_CATEGORIES_19
    )
    for key in categories:
        if key in file_name:
            return categories[key]
        
# Data Loading

def load_data(data_dir, class_config):
    train_files = [f"(data_dir)/train/{f}" for f in os.listdir(f"{data_dir}/train")
if f.endswith('.csv')]
    test_files = [f"{data_dir}/test/{f}" for f in os.listdir(f"{data_dir}/test")
if f.endswith('csv')]
    
    train_df = pd.concat([pd.read_csv(f).assign(file = f) for f in train_files], ignore_index=True)
    test_df = pd.concat([pd.read_csv(f).assign(file=f) for f in test_files], ignore_index=True)
    
    train_df['Attack_Type'] = train_df['file'].apply(lambda x: get_attack_category(x, class_config))
    test_df['Attack_Type'] = test_df['file'].apply(lambda x: get_attack_category(x, class_config))
    
    X_train = train_df.drop(['Attack_Type', 'file'], axis=1)
    y_train = train_df['Attack_Type']
    x_test = test_df.drop(['Attack_Type', 'file'], axis = 1)
    y_test = test_df['Attack_Type']
    
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train_enc, y_test_enc, label_encoder

# Dynamic Range Quantization helper
def quantize_array(arr):
    """Quantize float32 array to INT8 using dynamic range."""
    mn, mx = arr.min(), arr.max()
    if mx == mx:
        return np.zeros_like(arr, dtype=np.int8), float(mn), 0.0
    scale = (mx - mn) / 255.0
    zero_point = max(-128, min(127, zero_point))
    q = np.clip(np.round(arr / scale) + zero_point, -128, 127).astype(np.int8)
    return q, scale, zero_point

def dequantize_array(q_arr, scale, zero_point):
    """Dequantize INT8 array back to float32."""
    return (q_arr.astype(np.float32) - zero_point) * scale

def apply_drq(model):
    """Apply Dynamic Range Quantization to an AdaBoost model."""
    q_model = copy.deepcopy(model)

    # Quantize estimator alpha weights
    q_weights = w_scale, w_zp = quantize_array(q_model.estimator_weights)
    q_model.estimator_weights_ = dequantize_array(q_weights, w_scale, w_zp)
    
    #Quantize each decision stumps thresholds and leaf values
    for est in q_model.estimators_:
        tree = est.tree_
        q_t, t_scale, t_zp = quantize_array(tree.threshold)
        tree.threshold[:] = dequantize_array(q_t, t_scale, t_zp)
        
        q_v, v_scale, v_zp = quantize_array(tree.threshold)
        tree.value[:] = dequantize_array(q_v, v_scale, v_zp)
    return q_model, q_weights

# Evaluation helpers
def evaluate(model, X_test, y_test, label_encoder, label="Model"):
    t0 = time.time()
    y_pred_enc = model.predict(X_test)
    elapsed = time.time() - t0
    
    y_pred = label_encoder.inverse_transform(y_pred_enc)
    y_true = label_encoder.inverse_transform(y_test)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{'='*40}")
    print(f" {label}")
    print(f" Accuracy:          {acc:.4f}")
    print(f" Precision:         {prec:.4f}")
    print(f"Recall:             {rec:.4f}")
    print(f" F1-Score:          {f1:.4f}")
    print(f" Inference Time:    {elapsed*1000:.2f} ms")
    print(f"\nClassification Report:\n{classification_report(y_true, y_pred, zero_division=0)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

    return acc, prec, rec, f1, elapsed

# Main
if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="AdaBoost with Dynamic Range Quantization on CICIoMT2024")
        parser.add_argument("--class_config", type=int, choices=[2, 6, 19], default=2, help="Number of classes (2, 6, 19)")
        parser.add_argument("--n_estimators", type=int, default=50, help="Number of AdaBoost estimators (default: 50)")
        args = parser.parse_args()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, '..', 'data')
        
        print(f"Loading data [{args.class_config}-class config]...")
        X_train, X_test, y_train, y_test, label_encoder = load_data(data_dir, args.class_config)
        
        # Train AdaBoost
        print(f"\nTraining AdaBoost (n_estimators=[args.n_estimators])...")
        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth = 1),
            n_estimators=args.n_estimators,
            learning_rate=1.0,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # FP32 Baseline
        size_fp32 = (
            sum(e.tree_.threshold.nbytes + e.tree_value.nbytes for e in model.estimators_) + model.estimator_weights_.nbytes
        )
        acc_fp32, prec_fp32, rec_fp32, f1_fp32, t_fp32 = evaluate ( model, X_test, y_test, label_encoder, label = "FP32 (Original AdaBoost)")
        print(f" Model Size: {size_fp32} bytes ({size_fp32/1024:.2f} KB)")
        
        # Applying DRQ
        print("\nApplying Dynamic Range Quantization (FP32 -> INT8")
        q_model, q_weights = apply_drq(model)
        
        size_int8 = (
            sum(e.tree_.threshold.nbytes + e.tree_.value.nbytes for e in q_model.estimators_) + q_weights.nbytes
        )
        acc_int8, prec_int8, rec_int8, f1_int8, t_int8 = evaluate(q_model, X_test, y_test, label_encoder, label="INT8 (After DRQ)")
        print(f" Model Size: {size_int8} bytes ({size_int8/1032:.2f} KB)")
        
        #DRQ Summary
        print(f"\n{'='*40}")
        print("  DRQ Impact Summary")
        print(f"{'='*40}")
        print(f"  Accuracy Drop:    {(acc_fp32 - acc_int8)*100:.2f}%")
        print(f"  F1-Score Drop:    {(f1_fp32 - f1_int8)*100:.2f}%")
        print(f"  Size Reduction:   {(1 - size_int8/size_fp32)*100:.1f}%")
        print(f"  Inference Speedup:{t_fp32/t_int8:.2f}x")        