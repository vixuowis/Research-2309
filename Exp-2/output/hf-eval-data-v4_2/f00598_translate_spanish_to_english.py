# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_spanish_to_english(text: str) -> str:
    """Translate a given text from Spanish to English using the Helsinki-NLP/opus-mt-es-en model.

    Args:
        text (str): The text in Spanish to be translated.

    Returns:
        str: The translated text in English.

    Raises:
        ValueError: If the input text is not a string or is empty.
    """
    if not isinstance(text, str) or not text:
        raise ValueError('Input text must be a non-empty string.')
    translation_pipeline = pipeline('translation_es_to_en', model='Helsinki-NLP/opus-mt-es-en')
    result = translation_pipeline(text)
    return result[0]['translation_text']

# test_function_code --------------------

def test_translate_spanish_to_english():
    print("Testing started.")

    # Test case 1: Regular translation
    print("Testing case [1/3] started.")
    spanish_text = "Hola, ¿cómo estás?"
    expected_translation = "Hello, how are you?"
    assert translate_spanish_to_english(spanish_text) == expected_translation, f"Test case [1/3] failed: Expected {expected_translation}"

    # Test case 2: Handle empty string
    print("Testing case [2/3] started.")
    try:
        translate_spanish_to_english("")
        assert False, "Test case [2/3] failed: ValueError expected for empty input."
    except ValueError:
        assert True

    # Test case 3: Handle non-string input
    print("Testing case [3/3] started.")
    try:
        translate_spanish_to_english(None)
        assert False, "Test case [3/3] failed: ValueError expected for non-string input."
    except ValueError:
        assert True

    print("Testing finished.")

# call_test_function_line --------------------

test_translate_spanish_to_english()