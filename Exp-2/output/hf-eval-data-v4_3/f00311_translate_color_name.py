# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# function_code --------------------

def translate_color_name(color_name: str) -> str:
    """
    Translates the color name from English to Italian using mT5 model.

    Args:
        color_name (str): The color name in English to be translated.

    Returns:
        str: The translated color name in Italian.

    Raises:
        ValueError: If the input color_name is not a string.
    """
    if not isinstance(color_name, str):
        raise ValueError('Input color_name must be a string')

    model = MT5ForConditionalGeneration.from_pretrained('google/mt5-base')
    tokenizer = MT5Tokenizer.from_pretrained('google/mt5-base')
    inputs = tokenizer.encode(f'translate English to Italian: {color_name}', return_tensors='pt')
    outputs = model.generate(inputs, max_length=40, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# test_function_code --------------------

def test_translate_color_name():
    print("Testing started.")

    # Test case 1: Translate 'red'
    print("Testing case [1/3] started.")
    translated_color = translate_color_name('red')
    assert translated_color.lower() == 'rosso', f"Test case [1/3] failed: Expected 'rosso', got {translated_color}"

    # Test case 2: Translate 'blue'
    print("Testing case [2/3] started.")
    translated_color = translate_color_name('blue')
    assert translated_color.lower() == 'blu', f"Test case [2/3] failed: Expected 'blu', got {translated_color}"

    # Test case 3: Handle non-string input
    print("Testing case [3/3] started.")
    try:
        translate_color_name(123)
        assert False, "Test case [3/3] failed: ValueError exception was not raised."
    except ValueError:
        assert True

    print("Testing finished.")

# call_test_function_line --------------------

test_translate_color_name()