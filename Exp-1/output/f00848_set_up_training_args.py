from typing import *
from transformers import TrainingArguments

def set_up_training_args():
    """
    Set up standard training arguments
    """
    default_args = {
        "output_dir": "tmp",
        "evaluation_strategy": "steps",
        "num_train_epochs": 1,
        "log_level": "error",
        "report_to": "none",
    }

