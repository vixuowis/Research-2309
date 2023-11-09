from transformers import pipeline


def translate_english_to_italian(text):
    """
    This function translates English text to Italian using the Helsinki-NLP/opus-mt-en-it model from Hugging Face Transformers.
    
    Args:
    text (str): The English text to be translated.
    
    Returns:
    str: The translated Italian text.
    """
    # Create a translation pipeline using the specified model
    translator = pipeline('translation_en_to_it', model='Helsinki-NLP/opus-mt-en-it')
    
    # Translate the text to Italian
    italian_text = translator(text)[0]['translation_text']
    
    return italian_text