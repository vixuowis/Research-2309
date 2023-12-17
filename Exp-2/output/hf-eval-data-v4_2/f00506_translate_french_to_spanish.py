# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def translate_french_to_spanish(text: str) -> str:
    """Translate French text to Spanish using pretrained model.

    Args:
        text (str): The French text to be translated.

    Returns:
        str: The translated Spanish text.

    Raises:
        ValueError: If text is not a string or is empty.
    """
    if not text or not isinstance(text, str):
        raise ValueError('Input text must be a non-empty string.')

    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-es')
    model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-fr-es')
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# test_function_code --------------------

def test_translate_french_to_spanish():
    print("Testing started.")

    # Test case 1: Typical sentence
    print("Testing case [1/3] started.")
    input_text = "Bonjour, comment ça va?"
    output_text = translate_french_to_spanish(input_text)
    assert isinstance(output_text, str), f"Test case [1/3] failed: Output is not a string."
    assert output_text != input_text, f"Test case [1/3] failed: Translation not performed."

    # Test case 2: Empty string
    print("Testing case [2/3] started.")
    try:
        translate_french_to_spanish("")
        assert False, "Test case [2/3] failed: Empty string should raise ValueError."
    except ValueError as e:
        assert str(e) == "Input text must be a non-empty string.", f"Test case [2/3] failed: Wrong error message."

    # Test case 3: Non-string input
    print("Testing case [3/3] started.")
    try:
        translate_french_to_spanish(123)
        assert False, "Test case [3/3] failed: Non-string input should raise ValueError."
    except ValueError as e:
        assert str(e) == "Input text must be a non-empty string.", f"Test case [3/3] failed: Wrong error message."
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_french_to_spanish()