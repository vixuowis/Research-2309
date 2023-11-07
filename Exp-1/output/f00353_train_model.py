from typing import *
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

def train_model(model, train_dataset, eval_dataset, tokenizer, data_collator, compute_metrics):
    """Train the model using Seq2SeqTrainer

    Args:
        model (PreTrainedModel): The model to train
        train_dataset (Dataset): The training dataset
        eval_dataset (Dataset): The evaluation dataset
        tokenizer (PreTrainedTokenizer): The tokenizer used for encoding the inputs
        data_collator (DataCollator): The data collator used for batching the inputs
        compute_metrics (Callable): The function used to compute the metrics
    Returns:
        None
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir="my_awesome_opus_books_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
