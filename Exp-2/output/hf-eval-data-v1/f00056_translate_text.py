from transformers import pipeline


def translate_text(text, source_lang, target_lang):
    """
    This function translates a given text from one language to another using the PyTorch Transformers library.
    The function uses the 'facebook/nllb-200-distilled-600M' model, which is a pre-trained model for translating text between multiple languages.
    
    Parameters:
    text (str): The text to be translated.
    source_lang (str): The source language code.
    target_lang (str): The target language code.
    
    Returns:
    str: The translated text.
    """
    # Initialize the NLP translation pipeline with the model 'facebook/nllb-200-distilled-600M'
    translator = pipeline(f'translation_{source_lang}_to_{target_lang}', model='facebook/nllb-200-distilled-600M')
    # Translate the text
    translated_text = translator(text)
    return translated_text[0]['translation_text']