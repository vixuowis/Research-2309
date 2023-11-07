from typing import *
from transformers import TrainingArguments

def create_training_arguments(output_dir, per_device_train_batch_size, num_train_epochs, fp16, save_steps, logging_steps, learning_rate, weight_decay, save_total_limit, remove_unused_columns, push_to_hub):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        fp16=fp16,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_total_limit=save_total_limit,
        remove_unused_columns=remove_unused_columns,
        push_to_hub=push_to_hub
    )
