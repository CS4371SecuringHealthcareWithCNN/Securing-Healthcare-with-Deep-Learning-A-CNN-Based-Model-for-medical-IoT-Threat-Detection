import tensorflow as tf
import numpy as np


def quantize_model(model, representative_data):
    """
    Quantize a Keras model to TensorFlow Lite
    """
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # IMPORTANT: Disable tensor list lowering for LSTM support
    converter._experimental_lower_tensor_list_ops = False  # Add this line
    
    # Enable quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Add representative dataset for quantization
    def representative_dataset():
        X_train, _ = representative_data
        for i in range(min(100, len(X_train))):
            # Add batch dimension if needed
            sample = X_train[i].reshape(1, *X_train[i].shape)
            yield [sample.astype(np.float32)]
    
    converter.representative_dataset = representative_dataset
    
    # Set target specification (important for LSTM compatibility)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Default TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS     # Fallback to TF ops when needed
    ]
    
    # Convert the model
    quantized_model = converter.convert()
    
    return quantized_model


def _generate_representative_dataset(data):
    """Generate representative dataset for quantization."""
    if isinstance(data, tuple):
        X, _ = data
    else:
        X = data
    
    def representative_dataset():
        for i in range(min(100, len(X))):
            x = X[i:i+1]
            # Handle different input shapes
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            yield tf.constant(x, dtype=tf.float32)
    
    return representative_dataset()


def save_quantized_model(quantized_model, save_path):
    """Save quantized model to .tflite file."""
    with open(save_path, 'wb') as f:
        f.write(quantized_model)
    print(f"Quantized model saved to {save_path}")


def load_quantized_model(model_path):
    """Load a quantized TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def get_model_size(model_or_path):
    """Get model size in MB."""
    if isinstance(model_or_path, bytes):
        size_bytes = len(model_or_path)
    else:
        with open(model_or_path, 'rb') as f:
            size_bytes = len(f.read())
    return size_bytes / (1024 * 1024)


def evaluate_quantized_model(interpreter, X_test, y_test):
    """Evaluate quantized model accuracy."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    correct = 0
    total = len(X_test)
    
    for i in range(total):
        x = X_test[i:i+1]
        if len(input_details[0]['shape']) > len(x.shape):
            # Add batch dimension if needed
            x = x.reshape(1, *input_details[0]['shape'][1:])
        
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        predicted = np.argmax(output)
        actual = np.argmax(y_test[i]) if len(y_test[i].shape) > 0 else y_test[i]
        
        if predicted == actual:
            correct += 1
    
    accuracy = correct / total
    print(f"Quantized model accuracy: {accuracy:.4f}")
    return accuracy