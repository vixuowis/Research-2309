from typing import *
from transformers import set_seed

def generate_code(model, tokenizer, input_text):
    """Generate python code based on the input text.

    Args:
        model: The pre-trained model used for generation.
        tokenizer: The tokenizer used for tokenizing the input text.
        input_text: The input text to generate code from.

    Returns:
        str: The generated python code.
    """
    set_seed(42)
    model_inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
    generated_ids = model.generate(**model_inputs, do_sample=True)
    generated_code = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_code
