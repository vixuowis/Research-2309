# function_import --------------------

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# function_code --------------------

def translate_english_to_german(src_text):
    """
    Translates English text to German using the MBartForConditionalGeneration model.

    Args:
        src_text (str): The source text in English that needs to be translated.

    Returns:
        str: The translated text in German.

    Raises:
        OSError: If there is an issue with loading the pre-trained model or tokenizers.
    """
    try:
        model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50')
        tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50', src_lang='en_XX', tgt_lang='de_DE')
        translated_output = model.generate(**tokenizer(src_text, return_tensors='pt'))
        tgt_text = tokenizer.batch_decode(translated_output, skip_special_tokens=True)
        return tgt_text
    except OSError as e:
        print(f'Error: {e}')

# test_function_code --------------------

def test_translate_english_to_german():
    """
    Tests the translate_english_to_german function with some test cases.
    """
    assert translate_english_to_german('Hello, how are you?') is not None
    assert translate_english_to_german('This is a test sentence.') is not None
    assert translate_english_to_german('I love programming.') is not None
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_english_to_german()