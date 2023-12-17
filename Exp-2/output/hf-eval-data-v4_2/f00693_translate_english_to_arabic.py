# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_arabic(text: str) -> str:
    """
    Translates English text to Arabic using Hugging Face Transformers.

    Args:
        text (str): The English text to be translated.

    Returns:
        str: The translated Arabic text.

    Raises:
        ValueError: If the input text is empty.
    """
    # Raise an error if the text is empty
    if not text:
        raise ValueError('Input text cannot be empty')
    
    # Initialize the translation pipeline
    translation_pipeline = pipeline('translation_en_to_ar', model='Helsinki-NLP/opus-mt-en-ar')
    
    # Perform the translation
    translated_text = translation_pipeline(text)[0]['translation_text']
    
    return translated_text

# test_function_code --------------------

def test_translate_english_to_arabic():
    print("Testing started.")

    # Test case 1: Translate a simple English sentence
    print("Testing case [1/2] started.")
    result = translate_english_to_arabic('Hello World')
    assert result == 'مرحبا بالعالم', f"Test case [1/2] failed: Expected 'مرحبا بالعالم', got {result}"

    # Test case 2: Raise ValueError on empty string
    print("Testing case [2/2] started.")
    try:
        translate_english_to_arabic('')
        assert False, 'Test case [2/2] failed: ValueError not raised on empty input'
    except ValueError as e:
        assert str(e) == 'Input text cannot be empty', f"Test case [2/2] failed: Incorrect error message {e}"
    
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_english_to_arabic()