# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def translate_text_to_french(input_text):
    """
    Translate English text to French using the T5ForConditionalGeneration model.

    Args:
        input_text (str): The English text to translate.

    Returns:
        str: The translated French text.
    """
    tokenizer = T5Tokenizer.from_pretrained('t5-3b')
    model = T5ForConditionalGeneration.from_pretrained('t5-3b')
    text_to_translate = f'translate English to French: {input_text}'
    inputs = tokenizer.encode(text_to_translate, return_tensors='pt')
    outputs = model.generate(inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# test_function_code --------------------

def test_translate_text_to_french():
    print("Testing started.")
    # Test case 1: Valid English text
    print("Testing case [1/1] started.")
    english_text = "Introducing the new eco-friendly water bottle made of high-quality stainless steel with double-wall insulation to keep your drinks cool for 24 hours or hot for 12 hours."
    expected_start = "Pr\u00e9sentation"
    translated_text = translate_text_to_french(english_text)
    assert translated_text.startswith(expected_start), f"Test case [1/1] failed: Expected translation to start with {expected_start}, got {translated_text}"
    print("Testing finished.")

# Run the test function
test_translate_text_to_french()