# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_french(input_text):
    # Initialize translation model using Hugging Face pipeline
    translator = pipeline('translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr')
    # Translate English text to French
    translated_text = translator(input_text)
    # Return the translated text
    return translated_text[0]['translation_text']

# test_function_code --------------------

def test_translate_english_to_french():
    print("Testing started.")
    # Test case 1: Simple greeting
    input_text = "Hello, how are you?"
    expected_output = "Bonjour, comment vas-tu?"
    translated_text = translate_english_to_french(input_text)
    assert translated_text == expected_output, f"Test case failed: Expected '{expected_output}', but got '{translated_text}'"

    # Test cases with more input data can be added here

    print("Testing finished.")

# Run the test function
test_translate_english_to_french()