from typing import *
from transformers import TFMobileViTForImageClassification
import tensorflow as tf

def convert_to_tflite_model(model_ckpt):
    """
    Converts a MobileViT checkpoint to generate a TensorFlow Lite model.
    
    Args:
    - model_ckpt (str): The path to the MobileViT checkpoint.
    
    Returns:
    - None
    """
    model = TFMobileViTForImageClassification.from_pretrained(model_ckpt)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    tflite_filename = model_ckpt.split("/")[-1] + ".tflite"
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)
