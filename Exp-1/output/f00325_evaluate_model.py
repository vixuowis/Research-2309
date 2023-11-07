from typing import *
import math

def evaluate_model(trainer):
	eval_results = trainer.evaluate()
	return math.exp(eval_results['eval_loss'])
