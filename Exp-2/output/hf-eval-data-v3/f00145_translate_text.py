# function_import --------------------

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# function_code --------------------

def translate_text(english_text: str, target_lang: str) -> str:
    """
    Translates the given English text to the target language using the M2M100 model.

    Args:
        english_text (str): The text in English to be translated.
        target_lang (str): The target language code to translate the text into.

    Returns:
        str: The translated text in the target language.
    """
    model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
    tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')
    tokenizer.src_lang = 'en'
    encoded_input = tokenizer(english_text, return_tensors='pt')
    generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

# test_function_code --------------------

def test_translate_text():
    """
    Tests the translate_text function with some test cases.
    """
    assert translate_text('Hello, world!', 'fr') == 'Bonjour, monde!'
    assert translate_text('Good morning', 'es') == 'Buenos días'
    assert translate_text('How are you?', 'de') == 'Wie geht es dir?'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_text()