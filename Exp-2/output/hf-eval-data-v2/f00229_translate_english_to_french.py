# function_import --------------------

from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# function_code --------------------

def translate_english_to_french(english_contract_text):
    """
    Translates English contract text to French using the pre-trained 'google/mt5-base' model.

    Args:
        english_contract_text (str): The English contract text to be translated.

    Returns:
        str: The translated French text.
    """
    model = MT5ForConditionalGeneration.from_pretrained('google/mt5-base')
    tokenizer = MT5Tokenizer.from_pretrained('google/mt5-base')
    inputs = tokenizer.encode('translate English to French: ' + english_contract_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=1000, num_return_sequences=1)
    translated_french_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_french_text

# test_function_code --------------------

def test_translate_english_to_french():
    """
    Tests the translate_english_to_french function by translating a sample English text and checking if the output is a string.
    """
    english_text = 'This is a contract.'
    french_text = translate_english_to_french(english_text)
    assert isinstance(french_text, str), 'The output should be a string.'

# call_test_function_code --------------------

test_translate_english_to_french()