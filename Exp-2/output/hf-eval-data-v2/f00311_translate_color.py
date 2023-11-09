# function_import --------------------

from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# function_code --------------------

def translate_color(color_name):
    """
    Translates a color name from English to Italian using the google/mt5-base model from Hugging Face Transformers.

    Args:
        color_name (str): The name of the color in English.

    Returns:
        str: The name of the color in Italian.
    """
    model = MT5ForConditionalGeneration.from_pretrained('google/mt5-base')
    tokenizer = MT5Tokenizer.from_pretrained('google/mt5-base')

    inputs = tokenizer.encode(f'translate English to Italian: {color_name}', return_tensors='pt')
    outputs = model.generate(inputs, max_length=40, num_return_sequences=1)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# test_function_code --------------------

def test_translate_color():
    """
    Tests the translate_color function by translating a few sample colors and checking if the output is a non-empty string.
    """
    sample_colors = ['red', 'green', 'blue']
    for color in sample_colors:
        translated_color = translate_color(color)
        assert isinstance(translated_color, str)
        assert len(translated_color) > 0

# call_test_function_code --------------------

test_translate_color()