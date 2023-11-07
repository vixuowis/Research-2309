from typing import *
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

def train_model(checkpoint):
    
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
