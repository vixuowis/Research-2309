# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_spanish(text):
    """
    Translate English text to Spanish using a pretrained model.

    Args:
        text (str): The English text to be translated.

    Returns:
        str: The translated Spanish text.

    Raises:
        ValueError: If the input text is not a string or is empty.
    """
    if not isinstance(text, str) or not text:
        raise ValueError('Input text must be a non-empty string.')
    translation_pipeline = pipeline('translation_en_to_es', model='Helsinki-NLP/opus-mt-en-es')
    results = translation_pipeline(text)
    return results[0]['translation_text']

# test_function_code --------------------

def test_translate_english_to_spanish():
    print("Testing started.")
    # Testing case 1: Translate a simple greeting
    print("Testing case [1/3] started.")
    english_text = 'Hello, how are you?'
    spanish_text = translate_english_to_spanish(english_text)
    assert spanish_text == 'Hola, ¿cómo estás?', f"Test case [1/3] failed: Expected 'Hola, ¿cómo estás?' but got '{spanish_text}'."

    # Testing case 2: Testing with empty string input
    print("Testing case [2/3] started.")
    try:
        translate_english_to_spanish('')
        assert False, 'Test case [2/3] failed: ValueError was not raised for empty string.'
    except ValueError as e:
        assert str(e) == 'Input text must be a non-empty string.', f"Test case [2/3] failed: {e}."

    # Testing case 3: Testing with non-string input
    print("Testing case [3/3] started.")
    try:
        translate_english_to_spanish(None)
        assert False, 'Test case [3/3] failed: ValueError was not raised for non-string input.'
    except ValueError as e:
        assert str(e) == 'Input text must be a non-empty string.', f"Test case [3/3] failed: {e}."
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_english_to_spanish()