# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_spanish_to_english(text):
    """
    Translate a given text from Spanish to English using a pre-trained machine translation model.

    Parameters:
    text (str): The Spanish text to be translated.

    Returns:
    str: The translated English text.
    """
    # Initialize the translation pipeline
    translator = pipeline('translation_es_to_en', model='Helsinki-NLP/opus-mt-es-en')
    # Perform the translation
    result = translator(text)
    # Extract the translated text
    translated_text = result[0]['translation_text']
    return translated_text

# test_function_code --------------------

def test_translate_spanish_to_english():
    print("Testing translate_spanish_to_english function started.")
    # Test case: Translate a simple greeting from Spanish to English
    spanish_text = 'Hola, ¿cómo estás?'
    expected_translation = 'Hello, how are you?'
    print("Testing case [1/1] started.")
    translated_text = translate_spanish_to_english(spanish_text)
    assert translated_text.lower() == expected_translation.lower(), f"Test case [1/1] failed: Expected '{{expected_translation}}', got '{{translated_text}}'"
    print("Testing translate_spanish_to_english function finished.")

# Run the test function
test_translate_spanish_to_english()