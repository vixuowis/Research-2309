from typing import *
from transformers import TrainingArguments

def get_training_arguments():
    """Returns the TrainingArguments object with specified parameters."""
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        fp16=True,
        **default_args,
    )
    return training_args
