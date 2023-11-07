from typing import *
from transformers import AutoTokenizer

def load_preprocessing_class(class_name, model_name):
    return class_name.from_pretrained(model_name)
