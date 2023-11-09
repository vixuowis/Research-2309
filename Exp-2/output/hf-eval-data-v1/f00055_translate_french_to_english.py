from transformers import pipeline


def translate_french_to_english(text):
    """
    This function translates French text to English using the Helsinki-NLP/opus-mt-fr-en model from Hugging Face Transformers.
    
    Args:
    text (str): The French text to be translated.
    
    Returns:
    str: The translated English text.
    """
    # Create a translation pipeline using the specified model
    translation_pipeline = pipeline('translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
    
    # Use the pipeline to translate the text
    translated_text = translation_pipeline(text)
    
    # Return the translated text
    return translated_text[0]['translation_text']