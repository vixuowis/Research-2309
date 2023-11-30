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
    # set translation model parameters
    en_to_de_model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(en_to_de_model_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(en_to_de_model_name)
    # prepare the spanish text to be translated
    spanish_text = ">>es<< " + spanish_text
    input_ids = tokenizer(spanish_text, return_tensors="pt").input_ids
    # translate spanish text to polish text
    generated_tokens = model.generate(input_ids)
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return generated_text

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