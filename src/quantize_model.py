import tensorflow as tf
import numpy as np

def quantize_model(model):
    # model: compiled keras model

    #initialize converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # choose default optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # restrict weights to float16
    converter.target_spec.supported_types = [tf.float16]

    # convert
    qmodel = converter.convert()

    # save quantized model
    with open("model.tflite", "wb") as f:
        f.write(qmodel)
    return qmodel

def quantize_model_predict(tflite_model_bytes, X_sample):
    # tflite_model_bytes: raw model flatbuffer (binary)
    # X_sample: array of input samples

    # load binary into interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)

    # allocate memory for tensors
    interpreter.allocate_tensors()

    # size inspection for tensors
    for d in interpreter.get_tensor_details():
        try:
            tensor = interpreter.get_tensor(d["index"])
        except ValueError:
            continue  # skip non-materialized tensors

        if tensor.ndim < 2:
            continue  # skip scalars

        print(f"{d['name']:50s} | {tensor.dtype} | {tensor.nbytes} bytes")

    # retrieve metadata
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    results = []
    for sample in X_sample:
        # add a dimension to sample (expected by interpreter)
        # and cast to float 32 (expected by TFLite)
        interpreter.set_tensor(input_details[0]['index'], [sample.astype(np.float32)])

        #execute inference graph
        interpreter.invoke()

        # drop added dimension and add to results
        results.append(interpreter.get_tensor(output_details[0]['index'])[0])

    return np.array(results)