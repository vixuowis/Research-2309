from typing import *
from transformers import Trainer

def create_trainer(model_init, training_args, train_dataset, eval_dataset, compute_metrics, tokenizer, data_collator):
	"""
	Create a Trainer with the given parameters.

	Args:
	- model_init (Callable): Function that returns the model instance.
	- training_args (TrainingArguments): Training arguments.
	- train_dataset (Dataset): Training dataset.
	- eval_dataset (Dataset): Evaluation dataset.
	- compute_metrics (Callable): Function to compute evaluation metrics.
	- tokenizer (Tokenizer): Tokenizer instance.
	- data_collator (DataCollator): Data collator instance.

	Returns:
	- Trainer: The created Trainer instance.
	"""
	trainer = Trainer(
		model=None,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		compute_metrics=compute_metrics,
		tokenizer=tokenizer,
		model_init=model_init,
		data_collator=data_collator
	)

	return trainer

