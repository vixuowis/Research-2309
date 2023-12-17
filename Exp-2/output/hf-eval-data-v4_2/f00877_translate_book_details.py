# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_book_details(text, model='Helsinki-NLP/opus-mt-en-fr'):
    """
    Translate the given book details from English to French using a specified model.

    Args:
        text (str): The text in English to be translated.
        model (str): The model ID to use for translation (default: Helsinki-NLP/opus-mt-en-fr).

    Returns:
        str: The translated text in French.

    Raises:
        ValueError: If the text input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('The text input must be a string.')
    translator = pipeline('translation_en_to_fr', model=model)
    translated_text = translator(text)[0]['translation_text']
    return translated_text

# test_function_code --------------------

def test_translate_book_details():
    print("Testing started.")

    # Test case 1: Translate a simple English sentence
    print("Testing case [1/3] started.")
    english_text = "The Art of Computer Programming"
    french_text = translate_book_details(english_text)
    assert french_text and isinstance(french_text, str), f"Test case [1/3] failed: Expected a French string, got {french_text}"

    # Test case 2: Raise error on non-string input
    print("Testing case [2/3] started.")
    try:
        translate_book_details(None)
        assert False, "Test case [2/3] failed: ValueError not raised on non-string input."
    except ValueError:
        assert True

    # Test case 3: Use a different model
    print("Testing case [3/3] started.")
    alternative_model = 't5-small'
    french_text_alt = translate_book_details(english_text, model=alternative_model)
    assert french_text_alt and french_text != french_text_alt, f"Test case [3/3] failed: Translations are not expected to be the same with different models."
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_book_details()