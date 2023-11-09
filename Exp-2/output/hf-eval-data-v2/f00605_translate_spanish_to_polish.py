# function_import --------------------

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# function_code --------------------

def translate_spanish_to_polish(spanish_text):
    """
    This function translates Spanish text to Polish using the Hugging Face's transformers library.
    
    Args:
        spanish_text (str): The text in Spanish to be translated.
    
    Returns:
        str: The translated text in Polish.
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
    This function tests the translate_spanish_to_polish function by translating a Spanish text and checking if the output is a string.
    """
    spanish_text = 'Hola, ¿cómo estás?'
    polish_text = translate_spanish_to_polish(spanish_text)
    assert isinstance(polish_text, str), 'The output should be a string.'

# call_test_function_code --------------------

test_translate_spanish_to_polish()