# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_french_to_english(text):
    """
    Translate French text to English using a pre-trained model.

    Args:
        text (str): The French text to be translated.

    Returns:
        str: The translated English text.
    """
    # Initialize the translation model
    translator = pipeline('translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')

    # Translate the text
    translated_text = translator(text)[0]['translation_text']

    return translated_text

# test_function_code --------------------

def test_translate_french_to_english():
    print("Testing started.")

    # Sample French text
    sample_text = "Bonjour, comment ça va?"
    expected_translation = "Hello, how are you?"  # Expected result might vary slightly due to model differences

    # Test the translation function
    print("Testing sample text translation started.")
    translated_text = translate_french_to_english(sample_text)
    assert translated_text.lower() == expected_translation.lower(), f"Test failed: expected '{{expected_translation}}', got '{{translated_text}}'"
    print("Test passed: Sample text translation")

    print("Testing finished.")

# Run the test
if __name__ == "__main__":
    test_translate_french_to_english()