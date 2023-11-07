from typing import *
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

def generate_text(input_text, model_name, max_length=50):
    # Create tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForCausalLM.from_pretrained(model_name)

    # Encode input text
    input_ids = tokenizer.encode(input_text, return_tensors='tf')

    # Generate text
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    # Decode generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return generated text
    return generated_text
