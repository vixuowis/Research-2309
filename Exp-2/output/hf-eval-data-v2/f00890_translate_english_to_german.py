# function_import --------------------

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# function_code --------------------

def translate_english_to_german(src_text):
    """
    This function translates English text to German using the MBartForConditionalGeneration model.

    Args:
        src_text (str): The source text in English to be translated.

    Returns:
        str: The translated text in German.
    """
    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50')
    tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50', src_lang='en_XX', tgt_lang='de_DE')
    translated_output = model.generate(**tokenizer(src_text, return_tensors='pt'))
    tgt_text = tokenizer.batch_decode(translated_output, skip_special_tokens=True)
    return tgt_text[0]

# test_function_code --------------------

def test_translate_english_to_german():
    """
    This function tests the translate_english_to_german function by translating a sample English text and checking if the output is a string.
    """
    src_text = 'Here is the English material to be translated...'
    translated_text = translate_english_to_german(src_text)
    assert isinstance(translated_text, str), 'The output should be a string.'

# call_test_function_code --------------------

test_translate_english_to_german()