from typing import *
from transformers import Trainer, TrainingArguments


def train_model(model, train_dataset, eval_dataset, tokenizer, data_collator, compute_metrics):
    """Train a model using the given datasets and hyperparameters.

    Args:
        model (Model): The model to be trained.
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
        tokenizer (Tokenizer): The tokenizer.
        data_collator (DataCollator): The data collator.
        compute_metrics (Callable): The function to compute metrics.

    Returns:
        None
    """
    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
