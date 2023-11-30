# function_import --------------------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# function_code --------------------

def translate_french_to_spanish(input_text):
    """
    Translate French text to Spanish using Hugging Face Transformers.

    Args:
        input_text (str): The French text to be translated.

    Returns:
        str: The translated Spanish text.
    """
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-es")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "Helsinki-NLP/opus-mt-fr-es"
    )
    translated = model.generate(**tokenizer(input_text, return_tensors="pt"))
    output = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return output[0]

# test_function_code --------------------

def test_translate_french_to_spanish():
    """
    Test the function translate_french_to_spanish.
    """
    assert translate_french_to_spanish('Bonjour, comment ça va?') != ''
    assert translate_french_to_spanish('Je suis content de te voir.') != ''
    assert translate_french_to_spanish('Quel est votre nom?') != ''
    return 'All Tests Passed'


# call_test_function_code --------------------

test_translate_french_to_spanish()