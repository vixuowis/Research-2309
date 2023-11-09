from transformers import pipeline


def translate_text(text):
    """
    This function translates a given text into a specified language using the 'facebook/nllb-200-distilled-600M' model.
    The model is capable of translating text between 200 languages.
    The input language is automatically detected by the model.
    
    Parameters:
    text (str): The text to be translated.
    
    Returns:
    str: The translated text.
    """
    # Create a translation model using the 'facebook/nllb-200-distilled-600M' model
    translator = pipeline('translation_xx_to_yy', model='facebook/nllb-200-distilled-600M')
    
    # Translate the text
    translated_text = translator(text)
    
    return translated_text