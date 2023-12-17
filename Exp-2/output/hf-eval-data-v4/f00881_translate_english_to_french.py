# requirements_file --------------------

!pip install -U transformers optimum.onnxruntime

# function_import --------------------

from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# function_code --------------------

def translate_english_to_french(text):
    """
    Translate English text to French using the T5 translation model.

    Parameters:
        text (str): The English text to be translated.

    Returns:
        str: The translated text in French.
    """
    tokenizer = AutoTokenizer.from_pretrained('optimum/t5-small')
    model = ORTModelForSeq2SeqLM.from_pretrained('optimum/t5-small')
    translator = pipeline('translation_en_to_fr', model=model, tokenizer=tokenizer)
    results = translator(text)
    return results[0]['translation_text']

# test_function_code --------------------

def test_translate_english_to_french():
    print("Testing translation function.")

    # Test case 1: Simple sentence
    english_text1 = "Hello, how are you?"
    expected_french1 = "Bonjour, comment êtes-vous?"
    assert translate_english_to_french(english_text1) == expected_french1, f"Test case 1 failed."

    # Test case 2: Another simple sentence
    english_text2 = "I am a programmer."
    expected_french2 = "Je suis programmeur."
    assert translate_english_to_french(english_text2) == expected_french2, f"Test case 2 failed."

    print("All test cases passed.")

# Run the test function
test_translate_english_to_french()