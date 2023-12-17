# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_italian(text):
    """
    Translates English text to Italian using a pre-trained model.

    Args:
        text (str): The English text to be translated.

    Returns:
        str: The translated Italian text.

    Raises:
        ValueError: If the input text is not a string or is empty.
    """
    if not isinstance(text, str) or not text:
        raise ValueError('Input text must be a non-empty string.')
    translator = pipeline('translation_en_to_it', model='Helsinki-NLP/opus-mt-en-it')
    result = translator(text, return_text=True)
    italian_text = result[0]['translation_text']
    return italian_text

# test_function_code --------------------

def test_translate_english_to_italian():
    print("Testing started.")

    # Test case 1: translating a simple sentence
    print("Testing case [1/3] started.")
    input_text = 'Hello, world!'
    output = translate_english_to_italian(input_text)
    assert output == 'Ciao, mondo!', f"Test case [1/3] failed: {output}"

    # Test case 2: empty string should raise ValueError
    try:
        print("Testing case [2/3] started.")
        translate_english_to_italian('')
        assert False, "Test case [2/3] failed: ValueError not raised."
    except ValueError as e:
        assert str(e) == 'Input text must be a non-empty string.', f"Test case [2/3] failed: {e}"

    # Test case 3: non-string input should raise ValueError
    try:
        print("Testing case [3/3] started.")
        translate_english_to_italian(None)
        assert False, "Test case [3/3] failed: ValueError not raised."
    except ValueError as e:
        assert str(e) == 'Input text must be a non-empty string.', f"Test case [3/3] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_translate_english_to_italian()