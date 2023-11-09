# function_import --------------------

from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# function_code --------------------

def translate_english_to_french(text):
    """
    Translate English text to French using the 'optimum/t5-small' model from Transformers.

    Args:
        text (str): The English text to be translated.

    Returns:
        str: The translated French text.
    """
    tokenizer = AutoTokenizer.from_pretrained('optimum/t5-small')
    model = ORTModelForSeq2SeqLM.from_pretrained('optimum/t5-small')
    translator = pipeline('translation_en_to_fr', model=model, tokenizer=tokenizer)
    result = translator(text)
    return result[0]['translation_text']

# test_function_code --------------------

def test_translate_english_to_french():
    """
    Test the 'translate_english_to_french' function with a sample English text.
    """
    english_text = 'This is a story about a superhero who saves the day from evil villains.'
    french_text = translate_english_to_french(english_text)
    assert isinstance(french_text, str)

# call_test_function_code --------------------

test_translate_english_to_french()