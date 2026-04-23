import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_lr_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(num_classes, activation='softmax', input_shape=input_shape))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_lr_model(model, X_train, y_train_categorical, X_val, y_val_categorical, epochs=10, batch_size=1024):
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_categorical))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_categorical))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
    return model