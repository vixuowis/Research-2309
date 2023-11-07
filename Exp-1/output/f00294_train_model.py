from typing import *
from transformers import TrainingArguments, Trainer

def train_model(model, train_dataset, eval_dataset):
    # Define training hyperparameters
    training_args = TrainingArguments(
        output_dir="my_awesome_eli5_clm-model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=True,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Fine-tune the model
    trainer.train()
