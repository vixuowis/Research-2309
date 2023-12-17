# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_spanish_to_english(text):
    """
    Translate a given text from Spanish to English using pre-trained machine translation model.

    Args:
        text (str): A string in Spanish that needs to be translated to English.

    Returns:
        str: The translated text in English.

    Raises:
        ValueError: If text is not provided.
    """
    if not text:
        raise ValueError('Input text is required for translation.')
    translation_pipeline = pipeline('translation_es_to_en', model='Helsinki-NLP/opus-mt-es-en')
    result = translation_pipeline(text)
    return result[0]['translation_text']

# test_function_code --------------------

def test_translate_spanish_to_english():
    print('Testing started.')
    test_cases = [
        ('Hola, ¿Cómo estás?', 'Hello, how are you?'),
        ('Hasta la vista.', 'See you later.'),
        ('El gato está en la casa.', 'The cat is in the house.')
    ]

    for i, (input_text, expected_output) in enumerate(test_cases, start=1):
        print(f'Testing case [{i}/{len(test_cases)}] started.')
        translated_text = translate_spanish_to_english(input_text)
        assert translated_text.lower() == expected_output.lower(), f'Test case [{i}/{len(test_cases)}] failed: Expected {expected_output} but got {translated_text}.'
    print('Testing finished.')

# call_test_function_line --------------------

test_translate_spanish_to_english()