# function_import --------------------

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# function_code --------------------

def translate_spanish_to_polish(spanish_text):
    """
    Translate Spanish text to Polish using Hugging Face's MBartForConditionalGeneration model.

    Args:
        spanish_text (str): The Spanish text to be translated.

    Returns:
        str: The translated Polish text.
    """
    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    tokenizer.src_lang = 'es_ES'
    encoded_spanish = tokenizer(spanish_text, return_tensors='pt')
    generated_tokens = model.generate(**encoded_spanish, forced_bos_token_id=tokenizer.lang_code_to_id['pl_PL'])
    polish_subtitles = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return polish_subtitles[0]

# test_function_code --------------------

def test_translate_spanish_to_polish():
    """
    Test the function translate_spanish_to_polish.
    """
    spanish_text = 'Hola, ¿cómo estás?'
    polish_text = translate_spanish_to_polish(spanish_text)
    assert isinstance(polish_text, str), 'The result should be a string.'
    assert polish_text != '', 'The result should not be an empty string.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_spanish_to_polish()