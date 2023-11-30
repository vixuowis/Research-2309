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
    
    # Load pre-trained model and tokenizer
    
    try:
        tokenizer_src = MBartTokenizerFast.from_pretrained("facebook/mbart-large-cc25")
        model_src = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
    except OSError:
        print('Error! Issue with loading the pre-trained model or tokenizers.')
    
    # Translate source text to target language
    
    translation = model_src.generate(**tokenizer_src(src_text, return_tensors="pt").to("cuda"),max_length=512, num_beams=4)
    tgt_text = tokenizer_src.batch_decode(translation, skip_special_tokens=True)[0]
    
    return tgt_text

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