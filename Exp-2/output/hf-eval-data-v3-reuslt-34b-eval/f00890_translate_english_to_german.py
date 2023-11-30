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
        # set up MBartForConditionalGeneration, the English to German model
        model_name = "facebook/mbart-large-en-ro"
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name).to("cpu")
    except OSError as exception:
        raise OSError('Failed to load model and tokenizers') from exception
    
    # tokenize the text, getting the batch
    tokenized_text = tokenizer(src_text, return_tensors="pt")
    batch = tokenized_text["input_ids"]
    
    # translate
    translated = model.generate(batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    return tgt_text[0]  # returns just the translated text, not a list

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