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
    """
    model = AutoModel.from_pretrained('Helsinki-NLP/opus-mt-sv-en')
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-sv-en')
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# test_function_code --------------------

def test_translate_swedish_to_english():
    """
    Test the translate_swedish_to_english function with a sample Swedish text.
    """
    input_text = 'Stockholm är Sveriges huvudstad och största stad. Den har en rik historia och erbjuder många kulturella och historiska sevärdheter.'
    expected_output = 'Stockholm is the capital and largest city of Sweden. It has a rich history and offers many cultural and historical attractions.'
    assert translate_swedish_to_english(input_text) == expected_output

# call_test_function_code --------------------

test_translate_swedish_to_english()