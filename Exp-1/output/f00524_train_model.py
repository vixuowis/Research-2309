from typing import *
from transformers import TrainingArguments, Trainer

def train_model(model, train_dataset, eval_dataset):
	"""Train the model using the given datasets.

	Args:
		model (Model): The model to train.
		train_dataset (Dataset): The training dataset.
		eval_dataset (Dataset): The evaluation dataset.

	Returns:
		None
	"""
	training_args = TrainingArguments(
		output_dir="segformer-b0-scene-parse-150",
		learning_rate=6e-5,
		num_train_epochs=50,
		per_device_train_batch_size=2,
		per_device_eval_batch_size=2,
		save_total_limit=3,
		evaluation_strategy="steps",
		save_strategy="steps",
		save_steps=20,
		eval_steps=20,
		logging_steps=1,
		eval_accumulation_steps=5,
		remove_unused_columns=False,
		push_to_hub=True,
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		compute_metrics=compute_metrics,
	)

	trainer.train()
