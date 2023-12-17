# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def translate_russian_to_english(text):
    """Translate Russian text to English using the Helsinki-NLP/opus-mt-ru-en model.

    Args:
        text (str): The Russian text to be translated.

    Returns:
        str: The translated English text.

    Raises:
        ValueError: If the input text is not a string or is empty.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("Input text must be a non-empty string.")
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ru-en')
    model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-ru-en')
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# test_function_code --------------------

def test_translate_russian_to_english():
    print("Testing started.")
    sample_data = "привет"  # "привет" means "hello" in Russian

    # Test case 1: Valid Russian text
    print("Testing case [1/3] started.")
    translation = translate_russian_to_english(sample_data)
    assert translation.lower() == "hello", f"Test case [1/3] failed: expected 'hello', got '{translation}'"

    # Test case 2: Empty string input
    print("Testing case [2/3] started.")
    try:
        translate_russian_to_english("")
        assert False, "Test case [2/3] failed: ValueError not raised for empty string"
    except ValueError:
        pass  # Expected

    # Test case 3: Non-string input
    print("Testing case [3/3] started.")
    try:
        translate_russian_to_english(None)
        assert False, "Test case [3/3] failed: ValueError not raised for non-string input"
    except ValueError:
        pass  # Expected
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_russian_to_english()