# function_import --------------------

from transformers import MarianMTModel, MarianTokenizer

# function_code --------------------

def translate_catalan_to_spanish(catalan_text):
    """
    This function translates Catalan text into Spanish using the Helsinki-NLP/opus-mt-ca-es model from Hugging Face Transformers.

    Args:
        catalan_text (str): The Catalan text to be translated.

    Returns:
        str: The translated Spanish text.
    """
    model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ca-es')
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ca-es')

    tokenized_text = tokenizer.encode(catalan_text, return_tensors="pt")
    translated_tokens = model.generate(tokenized_text)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return translated_text

# test_function_code --------------------

def test_translate_catalan_to_spanish():
    """
    This function tests the translate_catalan_to_spanish function by translating a sample Catalan text and comparing it with the expected Spanish translation.
    """
    catalan_text = 'El text en català que vols traduir.'
    expected_spanish_translation = 'El texto en catalán que quieres traducir.'
    assert translate_catalan_to_spanish(catalan_text) == expected_spanish_translation

# call_test_function_code --------------------

test_translate_catalan_to_spanish()