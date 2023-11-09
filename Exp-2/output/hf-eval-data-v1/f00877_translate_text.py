from transformers import pipeline


def translate_text(input_text: str) -> str:
    """
    Translates English text to French using the Helsinki-NLP/opus-mt-en-fr model.

    Args:
        input_text (str): The text in English to be translated.

    Returns:
        str: The translated text in French.
    """
    translator = pipeline('translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr')
    translated_text = translator(input_text)
    return translated_text[0]['translation_text']