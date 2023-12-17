# requirements_file --------------------

import subprocess

requirements = ["transformers", "optimum.onnxruntime"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, Pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# function_code --------------------

def translate_property_description(description: str) -> str:
    """
    Translates a property description from English to French using a pretrained translation model.

    Args:
        description (str): The property description in English.

    Returns:
        str: The translated property description in French.

    Raises:
        ValueError: If the description is empty or not a string.
    """
    if not description:
        raise ValueError('The description must not be empty.')
    if not isinstance(description, str):
        raise ValueError('The description must be a string.')

    tokenizer = AutoTokenizer.from_pretrained('optimum/t5-small')
    model = ORTModelForSeq2SeqLM.from_pretrained('optimum/t5-small')
    translator = Pipeline('translation_en_to_fr', model=model, tokenizer=tokenizer)
    results = translator(description)
    return results[0]['translation_text']

# test_function_code --------------------

def test_translate_property_description():
    print("Testing started.")

    # Test case 1: Non-empty string
    print("Testing case [1/3] started.")
    non_empty_description = "Beautiful 3-bedroom house with a spacious garden and a swimming pool."
    assert translate_property_description(non_empty_description), "Test case [1/3] failed: The function should return a non-empty string."

    # Test case 2: Empty string
    print("Testing case [2/3] started.")
    empty_description = ""
    try:
        translate_property_description(empty_description)
        assert False, "Test case [2/3] failed: The function should raise a ValueError for empty description."
    except ValueError:
        pass

    # Test case 3: Non-string input
    print("Testing case [3/3] started.")
    non_string_description = 123
    try:
        translate_property_description(non_string_description)
        assert False, "Test case [3/3] failed: The function should raise a ValueError for non-string input."
    except ValueError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_translate_property_description()