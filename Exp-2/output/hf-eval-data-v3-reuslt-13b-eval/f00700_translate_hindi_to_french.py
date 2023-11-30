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
    model_name = 'facebook/mbart-large-50'
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    translator = MBartForConditionalGeneration.from_pretrained(model_name).cuda()
    batch = tokenizer([message], return_tensors="pt", padding=True, truncation=True).to('cuda') # batch is a dict with 3 keys
    generated_tokens = translator.generate(**batch)
    french = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_tokens]
    return french[0].capitalize() # returns a string

# function_import --------------------

from transformers import MT5ForConditionalGeneration, MT5TokenizerFast

# function_code --------------------


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