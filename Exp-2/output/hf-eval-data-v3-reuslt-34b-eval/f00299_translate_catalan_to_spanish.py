# function_import --------------------

from transformers import MarianMTModel, MarianTokenizer

# function_code --------------------

def translate_catalan_to_spanish(catalan_text):
    """
    Translate Catalan text to Spanish using Hugging Face's MarianMTModel.

    Args:
        catalan_text (str): The Catalan text to be translated.

    Returns:
        str: The translated Spanish text.
    """    
    model = 'Helsinki-NLP/opus-mt-ca-es'
    tokenizer = MarianTokenizer.from_pretrained(model)
    model = MarianMTModel.from_pretrained(model).to("cuda")

    input_ids = tokenizer(catalan_text, return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda")
    response = model.generate(input_ids)
    
    translated = [tokenizer.decode(t, skip_special_tokens=True) for t in response][0]
    return translated


# test_function_code --------------------

def test_translate_catalan_to_spanish():
    """
    Test the function translate_catalan_to_spanish.
    """
    catalan_text1 = 'El text en català que vols traduir.'
    catalan_text2 = 'Bona tarda, com estàs?'
    catalan_text3 = 'Estic bé, gràcies.'

    assert isinstance(translate_catalan_to_spanish(catalan_text1), str)
    assert isinstance(translate_catalan_to_spanish(catalan_text2), str)
    assert isinstance(translate_catalan_to_spanish(catalan_text3), str)

    print('All Tests Passed')


# call_test_function_code --------------------

test_translate_catalan_to_spanish()