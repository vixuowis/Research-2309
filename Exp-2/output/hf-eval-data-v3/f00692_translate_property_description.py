# function_import --------------------

from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# function_code --------------------

def translate_property_description(text):
    """
    Translate property descriptions from English to French using the T5 model.

    Args:
        text (str): The property description in English.

    Returns:
        str: The translated property description in French.
    """
    tokenizer = AutoTokenizer.from_pretrained('optimum/t5-small')
    model = ORTModelForSeq2SeqLM.from_pretrained('optimum/t5-small')
    translator = pipeline('translation_en_to_fr', model=model, tokenizer=tokenizer)
    results = translator(text)
    return results[0]['translation_text']

# test_function_code --------------------

def test_translate_property_description():
    """
    Test the function translate_property_description.
    """
    assert translate_property_description('Beautiful 3-bedroom house with a spacious garden and a swimming pool.') == 'Belle maison de 3 chambres avec un jardin spacieux et une piscine.'
    assert translate_property_description('Cozy 2-bedroom apartment in the city center with a great view.') == 'Appartement confortable de 2 chambres en centre-ville avec une superbe vue.'
    assert translate_property_description('Luxurious villa by the sea with a private beach.') == 'Villa luxueuse en bord de mer avec une plage privée.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_property_description()