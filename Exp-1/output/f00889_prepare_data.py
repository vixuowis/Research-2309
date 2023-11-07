from typing import *
from transformers import TapasTokenizer
import pandas as pd

def prepare_data(model_name, data, queries, answer_coordinates, answer_text):
    tokenizer = TapasTokenizer.from_pretrained(model_name)
    
    table = pd.DataFrame.from_dict(data)
    
    inputs = tokenizer(
        table=table,
        queries=queries,
        answer_coordinates=answer_coordinates,
        answer_text=answer_text,
        padding='max_length',
        return_tensors='pt',
    )
    
    return inputs
