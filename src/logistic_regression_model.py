import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_lr_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(num_classes, activation='softmax', input_shape=input_shape))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_lr_model(model, X_train, y_train_categorical, X_val, y_val_categorical, epochs=10, batch_size=32):
    model.fit(X_train, y_train_categorical, epochs=epochs, batch_size=batch_size, 
              validation_data=(X_val, y_val_categorical))
    return model