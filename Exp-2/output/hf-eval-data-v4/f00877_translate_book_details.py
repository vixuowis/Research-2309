# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_book_details(text, model='Helsinki-NLP/opus-mt-en-fr'):
    """
    Translate the book title and details from English to French.

    Args:
    text (str): The text to be translated, in English.
    model (str): The model to be used for translation. Defaults to 'Helsinki-NLP/opus-mt-en-fr'.

    Returns:
    str: The translated text in French.
    """
    translator = pipeline('translation_en_to_fr', model=model)
    translation = translator(text, max_length=512)
    return translation[0]['translation_text']

# test_function_code --------------------

def test_translate_book_details():
    print("Testing the translation of book details from English to French.")

    # Test case 1: Translating a simple book title
    print("Test case 1: Simple book title")
    original_text = 'Life of Pi'
    expected_translation = 'Vie de Pi' # Expected result might differ
    assert translate_book_details(original_text) == expected_translation, "Test case 1 failed: translation does not match expected output"

    # Test case 2: Translating book details
    print("Test case 2: Book details")
    original_text = 'A fascinating adventure novel by Yann Martel.'
    expected_translation = 'Un roman d'aventure fascinant de Yann Martel.' # Expected result might differ
    assert translate_book_details(original_text) == expected_translation, "Test case 2 failed: translation does not match expected output"

    print("All test cases passed.")

# Run the test function
test_translate_book_details()