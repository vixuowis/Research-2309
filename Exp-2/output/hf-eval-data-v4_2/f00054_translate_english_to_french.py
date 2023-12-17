# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_french(input_text):
    """
    Translates English text to French using a pre-trained model.

    Args:
        input_text (str): The English text to be translated.

    Returns:
        str: The translated text in French.

    Raises:
        ValueError: If the input text is not a string or is empty.
    """
    if not isinstance(input_text, str) or not input_text:
        raise ValueError('Input text must be a non-empty string.')
    translate = pipeline('translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr')
    translated_data = translate(input_text)
    return translated_data[0]['translation_text']

# test_function_code --------------------

def test_translate_english_to_french():
    print("Testing started.")

    # Testing case 1: Translate a simple greeting
    print("Testing case [1/3] started.")
    input_text = "Hello, how are you?"
    expected_result = "Bonjour, comment êtes-vous?"
    assert translate_english_to_french(input_text) == expected_result, f"Test case [1/3] failed for input '{input_text}'"

    # Testing case 2: Translate an empty string (should raise ValueError)
    print("Testing case [2/3] started.")
    input_text = ""
    try:
        translate_english_to_french(input_text)
        assert False, "Test case [2/3] failed: ValueError not raised for empty input."
    except ValueError as e:
        assert str(e) == "Input text must be a non-empty string.", f"Test case [2/3] failed: {e}"

    # Testing case 3: Translate using non-string input (should raise ValueError)
    print("Testing case [3/3] started.")
    input_text = 123
    try:
        translate_english_to_french(input_text)
        assert False, "Test case [3/3] failed: ValueError not raised for non-string input."
    except ValueError as e:
        assert str(e) == "Input text must be a non-empty string.", f"Test case [3/3] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_translate_english_to_french()