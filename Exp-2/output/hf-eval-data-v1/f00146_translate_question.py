from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def translate_question(input_text: str) -> str:
    """
    This function translates a given English text to German using the google/flan-t5-large model.
    
    Args:
    input_text (str): The English text to be translated.
    
    Returns:
    str: The translated German text.
    """
    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
    
    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    
    # Generate the translation
    outputs = model.generate(input_ids)
    
    # Decode the output to get the translated text
    translated_text = tokenizer.decode(outputs[0])
    
    return translated_text