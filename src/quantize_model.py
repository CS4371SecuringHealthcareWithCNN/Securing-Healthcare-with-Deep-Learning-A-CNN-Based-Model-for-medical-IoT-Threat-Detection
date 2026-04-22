import tensorflow as tf
import numpy as np

def quantize_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()

def quantize_model_predict(tflite_model_bytes, X_sample):
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], X_sample[0:1].astype(np.float32))
    interpreter.invoke()

    return interpreter.get_tensor(output_details[0]['index'])