# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch

# function_code --------------------

def translate_color_to_italian(color_name: str) -> str:
    """
    Translate a color name from English to Italian.
    
    Parameters:
    color_name (str): The color name in English.
    
    Returns:
    str: The translated color name in Italian.
    """
    model = MT5ForConditionalGeneration.from_pretrained('google/mt5-base')
    tokenizer = MT5Tokenizer.from_pretrained('google/mt5-base')
    inputs = tokenizer.encode(f'translate English to Italian: {color_name}', return_tensors='pt')
    outputs = model.generate(inputs, max_length=40, num_return_sequences=1)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

# test_function_code --------------------

def test_translate_color_to_italian():
    print("Testing translate_color_to_italian function.")
    # Testing case for the color 'red'
    print("Testing case [1/1] started.")
    color_name = 'red'
    translated_color = translate_color_to_italian(color_name)
    assert translated_color.lower() == 'rosso', f"Test case failed: {color_name} translated to {translated_color}, expected 'rosso'"
    print("Testing case [1/1] finished successfully.")

# Run the test function
test_translate_color_to_italian()