from transformers import pipeline


def translate_french_to_english(text):
    """
    This function translates French text to English using the Helsinki-NLP/opus-mt-fr-en model.
    
    Args:
    text (str): The text in French to be translated.
    
    Returns:
    str: The translated text in English.
    """
    # Initialize the translation model
    translator = pipeline('translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
    
    # Translate the text
    translated_text = translator(text)[0]['translation_text']
    
    return translated_text