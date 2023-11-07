from typing import *
from transformers import Seq2SeqTrainer

def instantiate_trainer(training_args, model, train_dataset, eval_dataset, data_collator, tokenizer):
    
    """
    Instantiate the Trainer object and pass the model, dataset, and data collator to it.
    
    Args:
        training_args (TrainingArguments): The training arguments.
        model (PreTrainedModel): The model to train.
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
        data_collator (DataCollator): The data collator to use.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
    """
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

