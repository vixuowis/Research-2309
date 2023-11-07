from typing import *
import math

def evaluate_model(trainer):
	"""
	Evaluate the model and calculate its perplexity.

	Args:
	- trainer: The `transformers.Trainer` object.

	Returns:
	- perplexity: The perplexity of the model.
	"""
	eval_results = trainer.evaluate()
	perplexity = math.exp(eval_results['eval_loss'])
	return perplexity

