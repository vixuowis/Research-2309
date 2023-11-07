from typing import *
from transformers import TrainingArguments, Trainer

def train_model(model, train_dataset, eval_dataset, tokenizer):
	"""
	Train the model.

	Args:
		model (object): The model to train.
		train_dataset (object): The training dataset.
		eval_dataset (object): The evaluation dataset.
		tokenizer (object): The tokenizer.

	Returns:
		None
	"""

	training_args = TrainingArguments(
		output_dir="my_awesome_mind_model",
		evaluation_strategy="epoch",
		save_strategy="epoch",
		learning_rate=3e-5,
		per_device_train_batch_size=32,
		gradient_accumulation_steps=4,
		per_device_eval_batch_size=32,
		num_train_epochs=10,
		warmup_ratio=0.1,
		logging_steps=10,
		load_best_model_at_end=True,
		metric_for_best_model="accuracy",
		push_to_hub=True,
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		tokenizer=tokenizer,
		compute_metrics=compute_metrics,
	)

	trainer.train()
