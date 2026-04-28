import os
import argparse
from data_loader import load_and_preprocess_data
from model import create_cnn_model, train_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a CNN for network intrusion detection.")
    parser.add_argument("--class_config", type=int, choices=[2, 6, 19], default=2,
                        help="Number of classes for classification (2, 6, or 19)")
    args = parser.parse_args()

    # Get the absolute path of the directory where this script is located 
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    # Construct the full path to your data directory
    data_dir = os.path.join(script_dir, '..', 'data') 

    # Pass data_dir to the function:
    X_train, X_val, X_test, y_train_categorical, y_val_categorical, y_test_categorical, label_encoder = load_and_preprocess_data(
        data_dir, args.class_config)  # Pass data_dir here 

    input_shape = (X_train.shape[1], 1) 
    model = create_cnn_model(input_shape, y_train_categorical.shape[1])

    import tensorflow as tf 
    if tf.test.gpu_device_name():
        print('GPU is available!')
    else:
        print('GPU is not available. Using CPU.')

    model = train_model(model, X_train, y_train_categorical, X_val, y_val_categorical)

    # Save data for compression
    import numpy as np
    model.save("baseline.keras")
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test_categorical)

    # Stratified calibration: take an equal number of samples from each class.
    # This guarantees coverage of minority attack types, which is critical for
    # int8 quantization to estimate activation ranges correctly across all classes.
    y_train_int = y_train_categorical.argmax(axis=1)
    samples_per_class = 250  # 250 * 19 classes = 4750 calibration samples
    calib_indices = []
    for c in range(y_train_categorical.shape[1]):
        class_indices = np.where(y_train_int == c)[0]
        # Take the first samples_per_class indices from this class.
        # If a class has fewer samples, take all of them.
        n_take = min(samples_per_class, len(class_indices))
        calib_indices.extend(class_indices[:n_take].tolist())
    calib_indices = np.array(calib_indices)
    np.save("X_calib.npy", X_train[calib_indices])
    print("Saved model and test data for compression.")
    print(f"Calibration set: {len(calib_indices)} samples ({samples_per_class} per class, where available)")

    loss, accuracy = model.evaluate(X_test, y_test_categorical)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred_categorical = model.predict(X_test)
    y_pred_encoded = y_pred_categorical.argmax(axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    y_test_decoded = label_encoder.inverse_transform(y_test_categorical.argmax(axis=1))

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