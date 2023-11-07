from typing import *
from transformers import TrainingArguments, Trainer, logging

def train_model(model, ds):
	logging.set_verbosity_error()
	training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)
	trainer = Trainer(model=model, args=training_args, train_dataset=ds)
	result = trainer.train()
	return result
