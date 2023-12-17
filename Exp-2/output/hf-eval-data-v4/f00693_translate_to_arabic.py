# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_to_arabic(text):
    """
    Translate English text to Arabic using the Helsinki-NLP/opus-mt-en-ar model from Hugging Face Transformers.

    Parameters:
        text (str): The English text to be translated.

    Returns:
        str: The translated Arabic text.
    """
    translator = pipeline('translation_en_to_ar', model='Helsinki-NLP/opus-mt-en-ar')
    translation = translator(text, max_length=512)
    translated_text = translation[0]['translation_text']
    return translated_text

# test_function_code --------------------

def test_translate_to_arabic():
    print("Testing translate_to_arabic function.")

    # Test case 1: Translate a simple greeting
    english_text = 'Hello World'
    expected_arabic_translation = 'مرحباً بالعالم'
    arabic_translation = translate_to_arabic(english_text)
    assert arabic_translation == expected_arabic_translation, f"Test case failed: Expected {expected_arabic_translation}, got {arabic_translation}"

    # Test case 2: Translate a complex sentence
    english_text = 'This is a test for the translation function.'
    arabic_translation = translate_to_arabic(english_text)
    assert isinstance(arabic_translation, str), "Test case failed: The result should be a string"

    print("All test cases passed successfully!")