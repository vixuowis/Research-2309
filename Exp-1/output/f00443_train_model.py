from typing import *
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

def train_model(id2label, label2id):
    num_labels = len(id2label)
    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
    )

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir='./logs',
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
