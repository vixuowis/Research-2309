from transformers import pipeline


def translate_english_to_german(text):
    """
    This function translates English text to German using the Hugging Face Transformers library.
    The model used for translation is 'sshleifer/tiny-marian-en-de'.
    
    Parameters:
    text (str): The English text to be translated.
    
    Returns:
    str: The translated German text.
    """
    # Create a translation pipeline
    translator = pipeline('translation_en_to_de', model='sshleifer/tiny-marian-en-de')
    # Translate the text
    translated_text = translator(text)
    # Return the translated text
    return translated_text[0]['translation_text']