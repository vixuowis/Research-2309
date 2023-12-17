# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_english_to_french(text: str) -> str:
    """
    Translate a given text from English to French using a machine translation model.

    Args:
        text (str): The English text to be translated.

    Returns:
        str: The translated French text.

    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    translation_pipeline = pipeline('translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr')
    translated_text = translation_pipeline(text)
    return translated_text[0]['translation_text']

# test_function_code --------------------

def test_translate_english_to_french():
    print("Testing started.")
    test_cases = {
        'Hello, how are you?': 'Bonjour, comment êtes-vous?',
        'My name is John.': 'Je m'appelle John.',
        'I am a programmer.': 'Je suis programmeur.'
    }
    for i, (english, expected_french) in enumerate(test_cases.items(), 1):
        print(f"Testing case [{i}/{len(test_cases)}] started.")
        translated_french = translate_english_to_french(english)
        assert translated_french == expected_french, f"Test case [{i}/{len(test_cases)}] failed: Translated '{{translated_french}}' != Expected '{{expected_french}}'"
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_english_to_french()