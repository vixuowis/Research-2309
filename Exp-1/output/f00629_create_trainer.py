from typing import *
from transformers import Trainer

def create_trainer(model, training_args, data_collator, train_dataset, tokenizer):
	trainer = Trainer(
		model=model,
		args=training_args,
		data_collator=data_collator,
		train_dataset=train_dataset,
		tokenizer=tokenizer
	)
	return trainer
