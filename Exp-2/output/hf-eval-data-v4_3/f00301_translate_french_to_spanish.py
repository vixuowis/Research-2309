# requirements_file --------------------

import subprocess

requirements = ["torch", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# function_code --------------------

def translate_french_to_spanish(text: str) -> str:
    """
    Translate a given text from French to Spanish using a pre-trained model.

    Args:
        text (str): The French text to translate.

    Returns:
        str: The translated text in Spanish.

    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')
    
    # Load the pre-trained model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-fr-es')
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-es')
    
    # Tokenize and translate the text
    tokenized_input = tokenizer(text, return_tensors='pt')
    translated_output = model.generate(**tokenized_input)
    
    # Decode the translated text
    output_text = tokenizer.decode(translated_output[0], skip_special_tokens=True)
    return output_text

# test_function_code --------------------

def test_translate_french_to_spanish():
    print("Testing started.")
    sample_text = 'Bonjour, comment ça va?'
    expected_translation = 'Hola, ¿cómo estás?'

    # Test case 1: Valid French text
    print("Testing case [1/3] started.")
    translated_text = translate_french_to_spanish(sample_text)
    assert translated_text == expected_translation, f"Test case [1/3] failed: Expected {expected_translation}, got {translated_text}"

    # Test case 2: Empty string
    print("Testing case [2/3] started.")
    try:
        translate_french_to_spanish('')
        assert False, "Test case [2/3] failed: ValueError not raised for empty string."
    except ValueError:
        pass

    # Test case 3: Non-string input
    print("Testing case [3/3] started.")
    try:
        translate_french_to_spanish(None)
        assert False, "Test case [3/3] failed: ValueError not raised for non-string input."
    except ValueError:
        pass
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_french_to_spanish()