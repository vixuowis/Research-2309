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

    # Load pre-trained model, tokenizer and config
    model_name = "Helsinki-NLP/opus-mt-fr-es"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Preprocess inputs and generate predictions
    preprocessed_input = tokenizer([input_text], return_tensors="pt")
    translated = model.generate(**preprocessed_input)
    spanish_output = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    # Return the translation
    return spanish_output[0]

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