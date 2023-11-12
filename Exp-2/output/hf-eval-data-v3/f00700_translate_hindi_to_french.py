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
    Tests the translate_hindi_to_french function with some example messages.
    """
    message1 = 'आपकी प्रेज़टेशन का आधार अच्छा था, लेकिन डेटा विश्लेषण पर ध्यान देना चाहिए।'
    message2 = 'मैं आपके सुझाव पर विचार करूंगा।'
    message3 = 'यह एक उत्कृष्ट प्रदर्शन था।'
    assert isinstance(translate_hindi_to_french(message1), str)
    assert isinstance(translate_hindi_to_french(message2), str)
    assert isinstance(translate_hindi_to_french(message3), str)
    print('All Tests Passed')

# call_test_function_code --------------------

test_translate_hindi_to_french()