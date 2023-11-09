from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def translate_french_to_spanish(text):
    """
    This function translates French text to Spanish using the Helsinki-NLP/opus-mt-fr-es model from Hugging Face Transformers.
    
    Args:
    text (str): The French text to be translated.
    
    Returns:
    str: The translated Spanish text.
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-es')
    model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-fr-es')
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt')
    
    # Generate the translated text
    outputs = model.generate(**inputs)
    
    # Decode the translated text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translated_text