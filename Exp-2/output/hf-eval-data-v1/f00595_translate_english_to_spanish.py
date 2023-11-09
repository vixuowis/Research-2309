from transformers import pipeline


def translate_english_to_spanish(text):
    """
    This function translates English text to Spanish using the Hugging Face Transformers library.
    It uses the 'Helsinki-NLP/opus-mt-en-es' model which is specifically designed for English to Spanish language pairs.
    
    Args:
    text (str): The English text to be translated.
    
    Returns:
    str: The translated Spanish text.
    """
    # Create a translation pipeline
    translation_pipeline = pipeline('translation_en_to_es', model='Helsinki-NLP/opus-mt-en-es')
    
    # Translate the text
    translated_text = translation_pipeline(text)
    
    return translated_text[0]['translation_text']