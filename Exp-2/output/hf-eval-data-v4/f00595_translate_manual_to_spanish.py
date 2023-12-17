# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_manual_to_spanish(user_manual_text):
    """
    Translate the user manual from English to Spanish.

    Parameters:
    user_manual_text (str): The English text of the user manual to be translated.

    Returns:
    str: The translated user manual in Spanish.
    """
    translation_pipeline = pipeline('translation_en_to_es', model='Helsinki-NLP/opus-mt-en-es')
    translated_manual = translation_pipeline(user_manual_text)[0]['translation_text']
    return translated_manual

# test_function_code --------------------

def test_translate_manual_to_spanish():
    print("Testing started.")
    sample_data = 'Hello, this is a sample user manual to be translated.'

    # Testing case 1: Translating English to Spanish
    print("Testing case [1/1] started.")
    expected_translation = 'Hola, este es un manual de usuario de muestra para ser traducido.'
    actual_translation = translate_manual_to_spanish(sample_data)
    assert expected_translation.lower() in actual_translation.lower(), f"Test case [1/1] failed: Expected '{expected_translation}', got '{actual_translation}'."
    print("Testing finished.")

# Run the test function
test_translate_manual_to_spanish()