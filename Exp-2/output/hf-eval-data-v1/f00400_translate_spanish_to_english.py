from transformers import pipeline


def translate_spanish_to_english(text):
    """
    This function translates Spanish text to English using the Helsinki-NLP/opus-mt-es-en model from the transformers library.
    
    Args:
    text (str): The Spanish text to be translated.
    
    Returns:
    str: The translated English text.
    """
    # Create a translation pipeline with the specified model
    translation = pipeline('translation_es_to_en', model='Helsinki-NLP/opus-mt-es-en')
    
    # Translate the text and return the result
    translated_text = translation(text)[0]['translation_text']
    return translated_text