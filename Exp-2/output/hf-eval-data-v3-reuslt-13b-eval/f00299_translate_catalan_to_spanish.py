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

    # Set the model name
    mname = 'Helsinki-NLP/opus-mt-ca-es'

    # Download and load the tokenizer and model from Hugging Face Transformer
    tokenizer = MarianTokenizer.from_pretrained(mname)
    model = MarianMTModel.from_pretrained(mname)
    model.eval()

    # Tokenize the input text
    tokens = tokenizer([catalan_text], return_tensors="pt")['input_ids']

    # Translate the tokens to Spanish and convert to a string
    translation = model.generate(tokens)
    output = [tokenizer.decode(t, skip_special_tokens=True) for t in translation]

    return "".join(output)

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