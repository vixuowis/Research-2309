from typing import *
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer

def train_model(checkpoint, id2label, label2id):
    # Load SegFormer with AutoModelForSemanticSegmentation, and pass the model the mapping between label ids and label classes:
    model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)

