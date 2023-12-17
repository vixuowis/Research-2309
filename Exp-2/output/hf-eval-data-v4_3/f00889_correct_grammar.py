# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def correct_grammar(text: str) -> str:
    """
    Corrects grammatical mistakes in the input text using a pre-trained model.

    Args:
        text (str): The text with potential grammatical errors.

    Returns:
        str: The corrected text with improved grammar.

    Raises:
        ValueError: If the input text is empty.
        RuntimeError: If the model fails to load or process the input text.
    """
    if not text:
        raise ValueError('Input text cannot be empty.')
    
    try:
        corrector = pipeline('text2text-generation', 'pszemraj/flan-t5-large-grammar-synthesis')
        results = corrector(text)
        corrected_text = results[0]['generated_text']
        return corrected_text
    except Exception as e:
        raise RuntimeError(f'Model failed with error: {e}')

# test_function_code --------------------

def test_correct_grammar():
    print("Testing started.")

    test_cases = [
        ('i can has cheezburger', 'I can have cheeseburger.'),
        ('he go to home', 'He goes home.'),
        ('they is friends', 'They are friends.')
    ]

    for i, (input_text, expected) in enumerate(test_cases, start=1):
        print(f"Testing case [{i}/{len(test_cases)}] started.")
        try:
            corrected = correct_grammar(input_text)
            assert corrected == expected, f"Test case [{i}/{len(test_cases)}] failed: expected '{{expected}}', got '{{corrected}}'"
        except Exception as e:
            assert False, f"Test case [{i}/{len(test_cases)}] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_correct_grammar()