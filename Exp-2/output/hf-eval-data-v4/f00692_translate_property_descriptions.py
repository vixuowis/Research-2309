# requirements_file --------------------

!pip install -U transformers optimum.onnxruntime

# function_import --------------------

from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# function_code --------------------

def translate_property_descriptions(text):
    """
    Translate property descriptions from English to French.

    Parameters:
        text (str): Description of the property in English.

    Returns:
        str: Translated description in French.
    """
    tokenizer = AutoTokenizer.from_pretrained('optimum/t5-small')
    model = ORTModelForSeq2SeqLM.from_pretrained('optimum/t5-small')
    translator = pipeline('translation_en_to_fr', model=model, tokenizer=tokenizer)
    result = translator(text)
    return result[0]['translation_text']

# test_function_code --------------------

def test_translate_property_descriptions():
    print("Testing started.")
    # Test case 1: Simple sentence
    english_description = "Cozy studio with an incredible city view."
    french_translation = translate_property_descriptions(english_description)
    assert french_translation == '...', f"Test case failed: Expected French translation, got {french_translation}"  # Replace ... with actual expected translation

    # Test case 2: Longer description
    english_description = "Spacious 4-bedroom villa with a private pool and garden."
    french_translation = translate_property_descriptions(english_description)
    assert french_translation == '...', f"Test case failed: Expected French translation, got {french_translation}"  # Replace ... with actual expected translation

    print("Testing finished.")