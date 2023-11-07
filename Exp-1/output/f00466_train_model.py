from typing import *
from transformers import AutoModelForCTC, TrainingArguments, Trainer

def train_model():
    # Load Wav2Vec2 with AutoModelForCTC. Specify the reduction to apply with the ctc_loss_reduction parameter. It is often better to use the average instead of the default summation.
    model = AutoModelForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )
