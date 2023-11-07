from typing import *
from transformers import TrainingArguments

def enable_gradient_checkpointing(training_args: TrainingArguments) -> TrainingArguments:
    """
    Enable gradient checkpointing in the Trainer.

    Args:
        training_args (TrainingArguments): The training arguments.

    Returns:
        TrainingArguments: The modified training arguments with gradient checkpointing enabled.
    """
    training_args.gradient_checkpointing = True
    return training_args
