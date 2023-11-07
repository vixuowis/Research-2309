from typing import *
from transformers import Trainer

def train_model(model, training_args, train_dataset, eval_dataset, compute_metrics):
	"""
	Train the model using the given arguments.

	Args:
		model: The model to train.
		training_args: Training arguments.
		train_dataset: Training dataset.
		eval_dataset: Evaluation dataset.
		compute_metrics: Function to compute evaluation metrics.

	Returns:
		trainer: The trained model trainer.
	"""
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		compute_metrics=compute_metrics,
	)
	return trainer
