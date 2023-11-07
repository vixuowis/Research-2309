from typing import *
import tensorflow as tf

def rescale_logits(logits, image):
    """Rescales the logits to the original image size and applies argmax on the class dimension.

    Args:
        logits (tf.Tensor): The logits tensor.
        image (tf.Tensor): The image tensor.

    Returns:
        tf.Tensor: The predicted segmentation tensor."""
    logits = tf.transpose(logits, [0, 2, 3, 1])

    upsampled_logits = tf.image.resize(
        logits,
        image.size[::-1],
    )

    pred_seg = tf.math.argmax(upsampled_logits, axis=-1)[0]
