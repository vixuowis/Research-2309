from typing import *
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

def train_model() -> None:
    # Train the model using the Trainer class
    # Parameters:
    #     model (AutoModelForMultipleChoice): The pre-trained model to be fine-tuned
    # Returns:
    #     None
    model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train the model
    trainer.train()
