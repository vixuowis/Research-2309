# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_spanish_to_english(text):
    """
    Translate Spanish to English using the Helsinki-NLP model from Transformers.

    Args:
        text (str): The Spanish text to be translated.

    Returns:
        str: The translated English text.
    """
    translation = pipeline('translation_es_to_en', model='Helsinki-NLP/opus-mt-es-en')
    translated_text = translation(text)[0]['translation_text']
    return translated_text

# test_function_code --------------------

def test_translate_spanish_to_english():
    print("Testing translate_spanish_to_english function.")
    # Test case 1: Basic greeting
    spanish_text = 'Hola, ¿cómo estás?'
    expected_translation = 'Hello, how are you?'
    assert translate_spanish_to_english(spanish_text) == expected_translation, "Test case 1 failed."

    # Test case 2: Complex sentence
    spanish_text = 'El clima es muy agradable hoy.'
    expected_translation = 'The weather is very pleasant today.'
    assert translate_spanish_to_english(spanish_text) == expected_translation, "Test case 2 failed."

    print("All tests passed!")

# Run the test function
test_translate_spanish_to_english()