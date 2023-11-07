from typing import *
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

def prepare_tf_dataset(dataset, batch_size, shuffle, tokenizer):
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tf_dataset = model.prepare_tf_dataset_for_classification(dataset=dataset, batch_size=batch_size, shuffle=shuffle, tokenizer=tokenizer)
    return tf_dataset
