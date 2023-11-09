from transformers import pipeline


def translate_french_to_english(french_text):
    """
    Translates French text to English using the Helsinki-NLP/opus-mt-fr-en model.

    Args:
        french_text (str): The text in French to be translated to English.

    Returns:
        str: The translated text in English.

    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(french_text, str):
        raise ValueError('Input must be a string')

    translator = pipeline('translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
    translated_text = translator(french_text)
    english_text = translated_text[0]['translation_text']

    return english_text