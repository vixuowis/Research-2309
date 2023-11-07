from typing import *
from transformers import Trainer

def create_trainer(model, args, train_dataset, eval_dataset, compute_metrics):
    """Create a Trainer object with the given parameters.

    Args:
        model (nn.Module): The model to be trained.
        args (TrainingArguments): The training arguments.
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
        compute_metrics (Callable): A function to compute evaluation metrics.

    Returns:
        Trainer: The created Trainer object.
    """
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    return trainer

