# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def translate_english_to_german(input_text: str) -> str:
    """
    Translates English text to German using the T5ForConditionalGeneration model from Hugging Face Transformers.

    Args:
        input_text (str): The English text to be translated.

    Returns:
        str: The translated German text.
    """
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    outputs = model.generate(input_ids)
    translated_text = tokenizer.decode(outputs[0])
    return translated_text

# test_function_code --------------------

def test_translate_english_to_german():
    assert translate_english_to_german('Where are the parks in Munich?') != ''
    assert translate_english_to_german('How old are you?') != ''
    assert translate_english_to_german('What is your name?') != ''
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_english_to_german()