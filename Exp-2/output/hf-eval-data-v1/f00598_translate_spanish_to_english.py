from transformers import pipeline


def translate_spanish_to_english(text):
    """
    This function translates Spanish text to English using the Helsinki-NLP/opus-mt-es-en model.
    
    Parameters:
    text (str): The Spanish text to be translated.
    
    Returns:
    str: The translated English text.
    """
    # Create an instance of the translation pipeline
    translation = pipeline('translation_es_to_en', model='Helsinki-NLP/opus-mt-es-en')
    # Use the pipeline to translate the text
    result = translation(text)
    # Extract the translated text from the result
    translated_text = result[0]['translation_text']
    return translated_text