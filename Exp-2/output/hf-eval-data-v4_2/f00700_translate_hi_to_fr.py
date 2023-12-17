# requirements_file --------------------

!pip install -U transformers==4.0.0

# function_import --------------------

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# function_code --------------------

def translate_hi_to_fr(message_hi):
    """
    Translates a message from Hindi to French using the mBART-50 model.

    Args:
        message_hi (str): The message in Hindi to be translated.

    Returns:
        str: The translated message in French.

    Raises:
        ValueError: If the input message is not a string.
    """
    
    if not isinstance(message_hi, str):
        raise ValueError("The input message must be a string.")

    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    tokenizer.src_lang = "hi_IN"
    encoded_hi = tokenizer(message_hi, return_tensors='pt')
    generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id['fr_XX'])
    translated_message = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return translated_message

# test_function_code --------------------

def test_translate_hi_to_fr():
    print("Testing started.")
    sample_message_hi = "आपकी प्रेज़टेशन का आधार अच्छा था, लेकिन डेटा विश्लेषण पर ध्यान देना चाहिये।"
    translated_message = translate_hi_to_fr(sample_message_hi)
    assert isinstance(translated_message, str), "Test case failed: The translated message is not a string."
    assert len(translated_message) > 0, "Test case failed: The translated message is empty."
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_hi_to_fr()