# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def translate_text_with_t5(input_text: str, source_language: str, target_language: str) -> str:
    """
    Translate text from source language to target language using T5 model.

    Args:
        input_text (str): The text to be translated.
        source_language (str): The source language code (e.g., 'en').
        target_language (str): The target language code (e.g., 'de').

    Returns:
        str: The translated text.

    Raises:
        ValueError: If source or target language is not supported.
    """
    # Check if the language codes are supported (you can add more languages as needed)
    if source_language not in ['en'] or target_language not in ['de', 'fr']:
        raise ValueError('Unsupported language code')

    # Model initialization
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')

    # Prepare the model input
    translation_task = f'translate {source_language} to {target_language}:' + input_text
    input_ids = tokenizer(translation_task, return_tensors='pt').input_ids

    # Generate the translated text
    outputs = model.generate(input_ids)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated_text

# test_function_code --------------------

def test_translate_text_with_t5():
    print("Testing started.")

    # Test case 1: Known source and target languages
    print("Testing case [1/2] started.")
    input_text = 'I have a doctor\'s appointment tomorrow morning.'
    source_language = 'en'
    target_language = 'de'
    expected_output = 'Ich habe morgen früh einen Arzttermin.'
    translated_text = translate_text_with_t5(input_text, source_language, target_language)
    assert translated_text == expected_output, f"Test case [1/2] failed: Expected {expected_output} but got {translated_text}"

    # Test case 2: Unsupported language
    print("Testing case [2/2] started.")
    try:
        translate_text_with_t5(input_text, 'en', 'es')
        assert False, "Test case [2/2] failed: ValueError was not raised for unsupported language"
    except ValueError:
        assert True
    
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_text_with_t5()