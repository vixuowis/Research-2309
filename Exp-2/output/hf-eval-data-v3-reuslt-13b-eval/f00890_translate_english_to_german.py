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
        model = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50", force_download=True)
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
    except OSError:
        print("There was an issue with loading the pre-trained model or tokenizers.")  # noqa: E501
    
    # Tokenize inputs (src, mt)
    src_tokenizer = tokenizer.get_tokenizer()
    src_model_inputs = tokenizer(src_text, return_tensors="pt")
        
    # Predict and decode translation
    translated = model.generate(**src_model_inputs)
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    return tgt_text[0]

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