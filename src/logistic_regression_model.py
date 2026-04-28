import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_lr_model(input_shape, num_classes):
    """Create and compile LR model""" 
    #create empty sequential model (linear stack)
    model = Sequential()

    # Dense applies (input x weights + bias) across all input features at once 
    # # number of neurons: num_classes 
    # # softmax: convert scores to probabilities 
    # # shape: matrix of samples X features
    model.add(Dense(num_classes, activation='softmax', input_shape=input_shape))

    # optimizer implements the Adam algorithm 
    # categorical_crossentropy: loss function to measure performance of model 
    # monitor accuracy during training
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_lr_model(model, X_train, y_train_categorical, X_val, y_val_categorical, epochs=10, batch_size=32):
    """Train LR model""" 
    # model: compiled model 
    # X_train: training feature data with shape (samples, features) 
    # y_train_categorical: training labels with shape (samples, num_classes), one-hot encoded 
    # X_val: validation data to monitor overfitting 
    # y_val_categorical: validation labels, one-hot encoded 
    # epochs: number of training cycles 
    # batch_size: number of samples analyzed at once 
    # fit weights using categorical_crossentropy over epochs 
    # update weights every batch 
    # evaluate validation data after each epoch
    model.fit(X_train, y_train_categorical, epochs=epochs, batch_size=batch_size, 
              validation_data=(X_val, y_val_categorical))
    return model

import os
import argparse
from data_loader import load_and_preprocess_data_lr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from quantize_model import quantize_model, quantize_model_predict

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
    X_train, X_val, X_test, y_train_categorical, y_val_categorical, y_test_categorical, label_encoder = load_and_preprocess_data_lr(
        data_dir, args.class_config)
    
    input_shape = (X_train.shape[1],)
    model = create_lr_model(input_shape, y_train_categorical.shape[1])

    if tf.test.gpu_device_name():
        print('GPU is available!')
    else:
        print('GPU is not available. Using CPU.')

    model = train_lr_model(model, X_train, y_train_categorical, X_val, y_val_categorical)
    model.save("model_original.keras")
    qmodel = quantize_model(model)

    print(f"Original model size: {os.path.getsize('model_original.keras') / 1024:.1f} KB")
    print(f"TFLite model size: {os.path.getsize('model.tflite') / 1024:.1f} KB")

    y_pred_categorical = model.predict(X_test)
    y_pred_encoded = y_pred_categorical.argmax(axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    y_test_decoded = label_encoder.inverse_transform(y_test_categorical.argmax(axis=1))

    accuracy = accuracy_score(y_test_decoded, y_pred)
    precision = precision_score(y_test_decoded, y_pred, average='weighted')
    recall = recall_score(y_test_decoded, y_pred, average='weighted')
    f1 = f1_score(y_test_decoded, y_pred, average='weighted')

    print("Original logistic regression model:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("\nClassification Report:\n", classification_report(y_test_decoded, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test_decoded, y_pred))

    y_pred_categorical = quantize_model_predict(qmodel, X_test)
    y_pred_encoded = y_pred_categorical.argmax(axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    y_test_decoded = label_encoder.inverse_transform(y_test_categorical.argmax(axis=1))

    accuracy = accuracy_score(y_test_decoded, y_pred)
    precision = precision_score(y_test_decoded, y_pred, average='weighted')
    recall = recall_score(y_test_decoded, y_pred, average='weighted')
    f1 = f1_score(y_test_decoded, y_pred, average='weighted')

    print("Compressed logistic regression model:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("\nClassification Report:\n", classification_report(y_test_decoded, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test_decoded, y_pred))