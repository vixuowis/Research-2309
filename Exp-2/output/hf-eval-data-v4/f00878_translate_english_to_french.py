# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_french(text):
    """
    Translate English text to French using a machine translation model.

    Parameters:
        text (str): The English text to translate.

    Returns:
        str: The translated French text.
    """
    # Initialize the translation pipeline with the model
    translation_pipeline = pipeline('translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr')
    # Translate the text
    result = translation_pipeline(text)
    # Extract the translated text from the result
    translated_text = result[0]['translation_text']
    return translated_text

# test_function_code --------------------

def test_translate_english_to_french():
    print("Testing translate_english_to_french function.")
    # Example English sentence
    english_sentence = "Hello, how are you?"
    # Expected French translation
    expected_translation = "Bonjour, comment êtes-vous ?"
    # Translate the English sentence
    actual_translation = translate_english_to_french(english_sentence)
    # Check if the actual translation matches the expected translation
    assert actual_translation == expected_translation, f"Test failed: Expected '" + expected_translation + "' but got '" + actual_translation + "'"
    print("Test passed.")

# Run the test function
test_translate_english_to_french()