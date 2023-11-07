from typing import *
from transformers import T5ForConditionalGeneration, T5Tokenizer

def translate_text(text: str) -> str:
    """
    Translate the given text from English to French using the T5 model.

    Args:
        text (str): The text to be translated.

    Returns:
        str: The translated text.
    """
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    prefix = 'translate English to French:'

    input_text = prefix + text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    output_ids = model.generate(input_ids)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text
