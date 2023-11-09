# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def translate_english_to_german(input_text: str) -> str:
    """
    Translates English text to German using the 'google/flan-t5-xl' model from Hugging Face Transformers.

    Args:
        input_text (str): The English text to be translated.

    Returns:
        str: The translated German text.
    """
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xl')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-xl')
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0])

# test_function_code --------------------

def test_translate_english_to_german():
    """
    Tests the translate_english_to_german function by translating a sample English text and checking if the output is a string.
    """
    input_text = 'How old are you?'
    translated_text = translate_english_to_german(input_text)
    assert isinstance(translated_text, str), 'The output should be a string.'

# call_test_function_code --------------------

test_translate_english_to_german()