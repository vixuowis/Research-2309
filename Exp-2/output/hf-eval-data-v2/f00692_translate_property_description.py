# function_import --------------------

from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# function_code --------------------

def translate_property_description(property_description):
    """
    Translates a property description from English to French using the pre-trained model 'optimum/t5-small'.

    Args:
        property_description (str): The property description in English.

    Returns:
        str: The translated property description in French.
    """
    tokenizer = AutoTokenizer.from_pretrained('optimum/t5-small')
    model = ORTModelForSeq2SeqLM.from_pretrained('optimum/t5-small')
    translator = pipeline('translation_en_to_fr', model=model, tokenizer=tokenizer)
    results = translator(property_description)
    return results[0]['translation_text']

# test_function_code --------------------

def test_translate_property_description():
    """
    Tests the function translate_property_description.
    """
    property_description = 'Beautiful 3-bedroom house with a spacious garden and a swimming pool.'
    translated_description = translate_property_description(property_description)
    assert isinstance(translated_description, str)

# call_test_function_code --------------------

test_translate_property_description()