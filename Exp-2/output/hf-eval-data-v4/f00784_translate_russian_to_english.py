# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def translate_russian_to_english(russian_text):
    """
    Translates Russian text to English using the pre-trained 'Helsinki-NLP/opus-mt-ru-en' model.
    
    Args:
        russian_text (str): The Russian text to be translated.
    
    Returns:
        str: The translated English text.
    """
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ru-en')
    model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-ru-en')
    inputs = tokenizer(russian_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# test_function_code --------------------

def test_translate_russian_to_english():
    print("Testing started.")
    
    russian_word = "привет"  # "Hello" in Russian
    expected_translation = "Hello"
    actual_translation = translate_russian_to_english(russian_word)
    assert expected_translation.lower() in actual_translation.lower(), f"Test case [1/3] failed: Expected '{expected_translation}', got '{actual_translation}'"

    russian_sentence = "Как дела?"  # "How are you?" in Russian
    expected_translation = "How are you?"
    actual_translation = translate_russian_to_english(russian_sentence)
    assert expected_translation.lower() in actual_translation.lower(), f"Test case [2/3] failed: Expected '{expected_translation}', got '{actual_translation}'"

    russian_paragraph = "Это большой успех для нашей компании."  # "This is a great success for our company." in Russian
    expected_translation = "This is a great success for our company."
    actual_translation = translate_russian_to_english(russian_paragraph)
    assert expected_translation.lower() in actual_translation.lower(), f"Test case [3/3] failed: Expected '{expected_translation}', got '{actual_translation}'"
    
    print("Testing finished.")

test_translate_russian_to_english()