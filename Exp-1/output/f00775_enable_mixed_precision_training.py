from typing import *
from transformers import TrainingArguments

def enable_mixed_precision_training(training_args: TrainingArguments) -> TrainingArguments:
    """
    Enable mixed precision training by setting the `fp16` flag to `True`.

    Args:
        training_args (TrainingArguments): The training arguments object.

    Returns:
        TrainingArguments: The updated training arguments object.
    """
    training_args.fp16 = True
    return training_args
