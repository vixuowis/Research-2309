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
    
    mbart = MBartForConditionalGeneration.from_pretrained("Helsinki-NLP/opus-mt-es-pl")
    tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-cc25')
    
    # encode text to input tokens for the model
    input_ids = tokenizer(spanish_text, return_tensors='pt').input_ids
    
    # generate output tokens for translation using model and input ids
    generated_tokens = mbart.generate(input_ids)
    
    # decode the tokens to text
    translated_polish_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    return translated_polish_text

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