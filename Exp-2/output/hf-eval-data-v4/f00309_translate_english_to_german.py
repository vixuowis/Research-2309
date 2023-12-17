# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_german(text):
    """
    Translate English text to German using the Hugging Face Transformers model.

    Parameters:
        text (str): The English text to be translated.

    Returns:
        str: The translated German text.
    """
    # Initialize the translation pipeline with the specific model
    translator = pipeline('translation_en_to_de', model='sshleifer/tiny-marian-en-de')

    # Perform the translation on the input text
    translated_text = translator(text, max_length=512)[0]['translation_text']

    return translated_text

# test_function_code --------------------

def test_translate_english_to_german():
    print("Testing translation function...")

    # Test case 1: Simple greeting
    english_text = "Hello, how are you?"
    expected_german_text = "Hallo, wie geht es dir?" # Expected translation may vary depending on the model accuracy
    translated_text = translate_english_to_german(english_text)
    assert translated_text == expected_german_text, f"Test case failed: expected {expected_german_text}, got {translated_text}"

    # Additional test cases with different types of sentences can be added here

    print("All tests passed!")

# Run the test function
test_translate_english_to_german()