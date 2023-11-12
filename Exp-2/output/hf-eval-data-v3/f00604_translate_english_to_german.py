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

    Raises:
        OSError: If there is an issue with loading the model or tokenizing the input.
    """
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xl')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-xl')
    input_text = 'translate English to German: ' + input_text
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0])

# test_function_code --------------------

def test_translate_english_to_german():
    """
    Tests the translate_english_to_german function with some example sentences.
    """
    assert translate_english_to_german('How old are you?') == 'Wie alt bist du?'
    assert translate_english_to_german('What is your name?') == 'Wie heißt du?'
    assert translate_english_to_german('Where do you live?') == 'Wo wohnst du?'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_english_to_german()