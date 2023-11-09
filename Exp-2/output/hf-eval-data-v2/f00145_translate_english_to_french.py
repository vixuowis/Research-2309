# function_import --------------------

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# function_code --------------------

def translate_english_to_french(english_text):
    """
    Translates English text to French using the M2M100ForConditionalGeneration model from Hugging Face Transformers.

    Args:
        english_text (str): The text in English to be translated.

    Returns:
        str: The translated text in French.
    """
    model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
    tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')
    tokenizer.src_lang = 'en'
    encoded_input = tokenizer(english_text, return_tensors='pt')
    generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id('fr'))
    french_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return french_text

# test_function_code --------------------

def test_translate_english_to_french():
    """
    Tests the function translate_english_to_french.
    """
    english_text = 'Welcome to our hotel, we hope you enjoy your stay.'
    french_text = translate_english_to_french(english_text)
    assert isinstance(french_text, str), 'The result should be a string.'
    assert french_text != '', 'The result should not be an empty string.'

# call_test_function_code --------------------

test_translate_english_to_french()