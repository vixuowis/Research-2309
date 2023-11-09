# function_import --------------------

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# function_code --------------------

def translate_hindi_to_french(message):
    """
    Translates a message from Hindi to French using the Hugging Face's MBartForConditionalGeneration model.

    Args:
        message (str): The message in Hindi to be translated.

    Returns:
        str: The translated message in French.
    """
    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    tokenizer.src_lang = 'hi_IN'
    encoded_hi = tokenizer(message, return_tensors='pt')
    generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id['fr_XX'])
    translated_message = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_message

# test_function_code --------------------

def test_translate_hindi_to_french():
    """
    Tests the translate_hindi_to_french function by translating a Hindi message and checking if the output is a string.
    """
    message = 'आपकी प्रेज़टेशन का आधार अच्छा था, लेकिन डेटा विश्लेषण पर ध्यान देना चाहिए।'
    translated_message = translate_hindi_to_french(message)
    assert isinstance(translated_message, str), 'The translated message should be a string.'

# call_test_function_code --------------------

test_translate_hindi_to_french()