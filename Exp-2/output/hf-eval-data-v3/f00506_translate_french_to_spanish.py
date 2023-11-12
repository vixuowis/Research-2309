# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def translate_french_to_spanish(text):
    """
    Translate French text to Spanish using Hugging Face Transformers.

    Args:
        text (str): The French text to be translated.

    Returns:
        str: The translated Spanish text.
    """
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-es')
    model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-fr-es')
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# test_function_code --------------------

def test_translate_french_to_spanish():
    assert translate_french_to_spanish('Bonjour, comment ça va?') != ''
    assert translate_french_to_spanish('Je suis content.') != ''
    assert translate_french_to_spanish('Il fait beau aujourd'hui.') != ''
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_french_to_spanish()