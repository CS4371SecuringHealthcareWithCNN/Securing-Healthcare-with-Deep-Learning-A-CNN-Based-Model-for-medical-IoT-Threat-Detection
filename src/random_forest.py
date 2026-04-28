import argparse
import os
import numpy as np
import tensorflow as tf
from data_loader import get_attack_category, ATTACK_CATEGORIES_2, ATTACK_CATEGORIES_6, ATTACK_CATEGORIES_19
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pandas as pd


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

converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('rf_model_compressed.tflite', 'wb') as f:
    f.write(tflite_model)

compressed_size = os.path.getsize('rf_model_compressed.tflite')
print(f"Compressed model size:  {compressed_size / 1024:.2f} KB")
print(f"Size reduction: {(1 - compressed_size / original_size) * 100:.2f}%")

interpreter = tf.lite.Interpreter(model_path='rf_model_compressed.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

preds = []
for i in range(len(rf_probs)):
    input_data = np.array([rf_probs[i]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    preds.append(np.argmax(output))

preds = label_encoder.inverse_transform(preds)
accuracy = accuracy_score(y_test_decoded, preds)
print(f"Compressed Model Accuracy: {accuracy: .4f}")
