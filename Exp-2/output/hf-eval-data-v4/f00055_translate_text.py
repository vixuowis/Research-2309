# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_text(text):
    # This function takes in a French text and translates it to English using a pre-trained model

    # Load the translation pipeline with the specific model
    translation_pipeline = pipeline('translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
    # Perform the translation
    translated_text = translation_pipeline(text)
    # Return the translated text
    return translated_text[0]['translation_text']

# test_function_code --------------------

def test_translate_text():
    print("Testing started.")
    # Sample French text
    sample_french_text = 'Le système éducatif français est composé d'écoles maternelles, d'écoles élémentaires, de collèges et de lycées.'
    # Expected English translation
    expected_english_translation = 'The French educational system is composed of nursery schools, elementary schools, middle schools, and high schools.'

    # Test case
    print("Testing translation started.")
    translated_text = translate_text(sample_french_text)
    assert translated_text == expected_english_translation, f"Test failed: Expected '{{expected_english_translation}}', got '{{translated_text}}'"
    print("Testing translation finished.")

# Execute the test function
test_translate_text()