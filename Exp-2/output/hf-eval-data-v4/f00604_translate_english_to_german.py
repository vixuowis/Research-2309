# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def translate_english_to_german(text):
    """
    Translate English text to German using the FLAN-T5 model.

    Args:
    text (str): The English text to be translated.

    Returns:
    str: The translated German text.
    """
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xl')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-xl')
    input_text = f'translate English to German: {text}'
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0])

# test_function_code --------------------

def test_translate_english_to_german():
    print("Testing started.")

    # Test case 1: Translate a simple greeting
    print("Testing case [1/3] started.")
    german_text = translate_english_to_german("Hello, how are you?")
    assert german_text == 'Hallo, wie geht es dir?', f"Test case [1/3] failed: Expected 'Hallo, wie geht es dir?' but got '{german_text}'"

    # Test case 2: Translate a question
    print("Testing case [2/3] started.")
    german_text = translate_english_to_german("What is your name?")
    assert german_text.startswith('Wie hei\u00DFt du?'), f"Test case [2/3] failed: Expected sentence starting with 'Wie hei\u00DFt du?' but got '{german_text}'"

    # Test case 3: Translate a complex sentence
    print("Testing case [3/3] started.")
    german_text = translate_english_to_german("The weather is nice today, isn't it?")
    assert 'Das Wetter ist heute sch\u00F6n, nicht wahr?' in german_text, f"Test case [3/3] failed: Expected a sentence containing 'Das Wetter ist heute sch\u00F6n, nicht wahr?' but got '{german_text}'"

    print("Testing finished.")

# Run test function
test_translate_english_to_german()