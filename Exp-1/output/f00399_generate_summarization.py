from typing import *
from transformers import TFAutoModelForSeq2SeqLM

def generate_summarization(inputs):
    model = TFAutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
    return outputs[0]['generated_text']
