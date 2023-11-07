from typing import *
from transformers import Trainer, TrainingArguments, DataCollatorForMultipleChoice


def train_model(model, train_dataset, eval_dataset, tokenizer):
	"""Train the model using the given datasets and tokenizer."
	
	:param model: The model to train.
	:param train_dataset: The training dataset.
	:param eval_dataset: The evaluation dataset.
	:param tokenizer: The tokenizer to use.
	:return: None
	"""
	
	training_args = TrainingArguments(
		output_dir="my_awesome_swag_model",
		evaluation_strategy="epoch",
		save_strategy="epoch",
		load_best_model_at_end=True,
		learning_rate=5e-5,
		per_device_train_batch_size=16,
		per_device_eval_batch_size=16,
		num_train_epochs=3,
		weight_decay=0.01,
		push_to_hub=True,
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		tokenizer=tokenizer,
		data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
		compute_metrics=compute_metrics,
	)

	trainer.train()
