# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_en_to_de(text: str) -> str:
    """
    Translate a given English text to German using Hugging Face Transformers.

    Args:
        text (str): The English text to be translated.

    Returns:
        str: The translated German text.

    Raises:
        ValueError: If the input text is empty.
    """
    if not text:
        raise ValueError('Input text cannot be empty.')
    translator = pipeline('translation_en_to_de', model='sshleifer/tiny-marian-en-de')
    translation = translator(text)
    return translation[0].get('translation_text', '')


# test_function_code --------------------

def test_translate_en_to_de():
    print("Testing started.")

    # Test case 1: Valid English text
    print("Testing case [1/3] started.")
    english_text = 'Hello, world!'
    assert translate_en_to_de(english_text) != '', f"Test case [1/3] failed: Translation of a valid text was empty."

    # Test case 2: Empty text
    print("Testing case [2/3] started.")
    empty_text = ''
    try:
        translate_en_to_de(empty_text)
        assert False, f"Test case [2/3] failed: ValueError was not raised for empty text."
    except ValueError:
        assert True

    # Test case 3: None as input (should raise an error)
    print("Testing case [3/3] started.")
    none_text = None
    try:
        translate_en_to_de(none_text)
        assert False, f"Test case [3/3] failed: ValueError was not raised for None input."
    except TypeError:
        assert True
    print("Testing finished.")


# call_test_function_line --------------------

test_translate_en_to_de()