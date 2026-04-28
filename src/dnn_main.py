import os
import argparse
import numpy as np
from data_loader import load_and_preprocess_data
from dnn_model import create_model, train_model
from dnn_quantize_model import quantize_model, save_quantized_model, get_model_size
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a DNN for network intrusion detection.")
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

    # DNN uses flattened input
    input_shape = (X_train.shape[1],)
    model = create_model(input_shape, y_train_categorical.shape[1])

    import tensorflow as tf 
    if tf.test.gpu_device_name():
        print('GPU is available!')
    else:
        print('GPU is not available. Using CPU.')
    
    # Saving the trained model to prevent re-training
    model_save_path = os.path.join(script_dir, f"dnn_baseline_{args.class_config}class.keras")
    
    if os.path.exists(model_save_path):
        print(f"Loading saved model from {model_save_path}...")
        model = tf.keras.models.load_model(model_save_path)
    else:
        model = train_model(model, X_train, y_train_categorical, X_val, y_val_categorical)
        model.save(model_save_path)
        print(f"Model saved for {model_save_path}")
        
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

    # Quantize the trained model
    print("\n--- Quantizing Model ---")
    quantized_model = quantize_model(model, representative_data=(X_train, y_train_categorical))
    
    # Save quantized model
    model_path = os.path.join(script_dir, '..', 'models')
    os.makedirs(model_path, exist_ok=True)
    quantized_path = os.path.join(model_path, f'model_{args.class_config}classes.tflite')
    save_quantized_model(quantized_model, quantized_path)
    
    # Report size reduction
    original_size = model.count_params() * 4 / (1024 * 1024)  # Approximate float32 size
    quantized_size = get_model_size(quantized_model)
    print(f"\nOriginal model parameters: {model.count_params():,}")
    print(f"Estimated original size: ~{original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")

    # Test the quantized model
    print("\n--- Testing Quantized Model ---")
    from dnn_quantize_model import load_quantized_model
    interpreter = load_quantized_model(quantized_path)
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Run inference on test set
    y_pred_quantized = []
    expected_shape = input_details[0]['shape']  # Full shape including batch
    
    for i in range(len(X_test)):
        x = X_test[i].astype(np.float32)
        # Reshape to match expected input shape (include batch dimension)
        if x.shape != tuple(expected_shape[1:]):
            x = x.reshape(expected_shape[1:])
        # Add batch dimension: (features,) -> (1, features)
        x = np.expand_dims(x, axis=0)
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        y_pred_quantized.append(np.argmax(output[0]))
    
    y_pred_quantized = np.array(y_pred_quantized)
    y_test_labels = y_test_categorical.argmax(axis=1)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    q_accuracy = accuracy_score(y_test_labels, y_pred_quantized)
    q_precision = precision_score(y_test_labels, y_pred_quantized, average='weighted')
    q_recall = recall_score(y_test_labels, y_pred_quantized, average='weighted')
    q_f1 = f1_score(y_test_labels, y_pred_quantized, average='weighted')
    
    print(f"Quantized Model Accuracy: {q_accuracy:.4f}")
    print(f"Quantized Model Precision: {q_precision:.4f}")
    print(f"Quantized Model Recall: {q_recall:.4f}")
    print(f"Quantized Model F1-Score: {q_f1:.4f}")