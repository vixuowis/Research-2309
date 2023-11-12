# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def translate_swedish_to_english(input_text: str) -> str:
    """
    Translate Swedish text to English using the Helsinki-NLP/opus-mt-sv-en model from Hugging Face Transformers.

    Args:
        input_text (str): The Swedish text to be translated.

    Returns:
        str: The translated English text.

    Raises:
        OSError: If there is an issue with loading the model or tokenizing the input.
    """
    model = AutoModel.from_pretrained('Helsinki-NLP/opus-mt-sv-en')
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-sv-en')
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs)
    translated_text = tokenizer.decode(outputs[0])
    return translated_text

# test_function_code --------------------

def test_translate_swedish_to_english():
    """Tests for the `translate_swedish_to_english` function"""
    assert translate_swedish_to_english('Stockholm är Sveriges huvudstad och största stad. Den har en rik historia och erbjuder många kulturella och historiska sevärdheter.') == 'Stockholm is the capital and largest city of Sweden. It has a rich history and offers many cultural and historical attractions.'
    assert translate_swedish_to_english('Jag älskar att resa.') == 'I love to travel.'
    assert translate_swedish_to_english('Välkommen till Sverige!') == 'Welcome to Sweden!'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_swedish_to_english()