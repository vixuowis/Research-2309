# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def translate_english_to_french(input_text):
    """
    Translates English text to French using the T5ForConditionalGeneration model from Hugging Face Transformers.

    Args:
        input_text (str): The English text to be translated.

    Returns:
        str: The translated French text.
    """
    tokenizer = T5Tokenizer.from_pretrained('t5-3b')
    model = T5ForConditionalGeneration.from_pretrained('t5-3b')
    inputs = tokenizer.encode('translate English to French: ' + input_text, return_tensors='pt')
    outputs = model.generate(inputs)
    translated_text = tokenizer.decode(outputs[0])
    return translated_text

# test_function_code --------------------

def test_translate_english_to_french():
    """
    Tests the translate_english_to_french function by translating a sample English text and checking if the output is a non-empty string.
    """
    sample_text = 'Introducing the new eco-friendly water bottle made of high-quality stainless steel with double-wall insulation to keep your drinks cool for 24 hours or hot for 12 hours.'
    translated_text = translate_english_to_french(sample_text)
    assert isinstance(translated_text, str)
    assert len(translated_text) > 0

# call_test_function_code --------------------

test_translate_english_to_french()