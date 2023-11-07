from typing import *
import tensorflow as tf

def run_xla_forward_pass(model, random_inputs):
	"""
	Run the forward pass with an XLA-compiled function.

	Args:
		model: The model to run the forward pass on.
		random_inputs: The random inputs to use for the forward pass.

	Returns:
		None
	"""
	
	xla_fn = tf.function(model, jit_compile=True)
	_ = xla_fn(random_inputs)

