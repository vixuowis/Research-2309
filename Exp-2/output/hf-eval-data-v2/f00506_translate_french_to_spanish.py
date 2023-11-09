# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def translate_french_to_spanish(text):
    """
    Translates French text to Spanish using the Helsinki-NLP/opus-mt-fr-es model from Hugging Face Transformers.

    Args:
        text (str): The French text to be translated.

    Returns:
        str: The translated Spanish text.
    """
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-es')
    model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-fr-es')
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# test_function_code --------------------

def test_translate_french_to_spanish():
    """
    Tests the translate_french_to_spanish function with a sample French text.
    """
    french_text = 'Bonjour, comment ça va?'
    translated_text = translate_french_to_spanish(french_text)
    assert isinstance(translated_text, str)

# call_test_function_code --------------------

test_translate_french_to_spanish()