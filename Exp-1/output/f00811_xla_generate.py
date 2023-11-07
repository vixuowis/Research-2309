from typing import *
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

def xla_generate(input_ids, **kwargs):
    """Generate sequences using XLA acceleration.

    Args:
        input_ids (tf.Tensor): Input tensor of shape (batch_size, sequence_length).
        **kwargs: Additional keyword arguments to pass to `model.generate()`.

    Returns:
        tf.Tensor: Output tensor of shape (batch_size, generated_sequence_length).
    """
    input_ids = tf.convert_to_tensor(input_ids)

    # Enable XLA
    tf.config.optimizer.set_jit(True)

    # Generate sequences
    generated_sequences = model.generate(input_ids, **kwargs)

    # Disable XLA
    tf.config.optimizer.set_jit(False)

    return generated_sequences
