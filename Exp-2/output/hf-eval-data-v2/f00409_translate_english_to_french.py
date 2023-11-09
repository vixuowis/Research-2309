# function_import --------------------

from transformers import T5ForConditionalGeneration, AutoTokenizer

# function_code --------------------

def translate_english_to_french(input_text: str) -> str:
    """
    Translates an English text to French using the T5ForConditionalGeneration model from Hugging Face Transformers.

    Args:
        input_text (str): The English text to be translated.

    Returns:
        str: The translated French text.
    """
    model = T5ForConditionalGeneration.from_pretrained('google/byt5-small')
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
    input_ids = tokenizer.encode(f"translate English to French: {input_text}", return_tensors="pt")
    output_ids = model.generate(input_ids)
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translated_text

# test_function_code --------------------

def test_translate_english_to_french():
    """
    Tests the translate_english_to_french function by translating a sample English text and checking if the output is a non-empty string.
    """
    input_text = 'Life is like a box of chocolates.'
    translated_text = translate_english_to_french(input_text)
    assert isinstance(translated_text, str)
    assert len(translated_text) > 0

# call_test_function_code --------------------

test_translate_english_to_french()