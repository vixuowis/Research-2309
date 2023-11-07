from typing import *
from transformers import TrainingArguments, Trainer

def train_model(model, training_args, train_dataset, eval_dataset, tokenizer, data_collator):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()
