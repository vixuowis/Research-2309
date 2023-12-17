# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def translate_english_to_german(sentence: str) -> str:
    """
    Translate an English sentence to German using a pre-trained T5 model.

    Args:
        sentence (str): The English sentence to be translated.

    Returns:
        str: The translated German sentence.
    """
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xl')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-xl')
    input_text = f'translate English to German: {sentence}'
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# test_function_code --------------------

def test_translate_english_to_german():
    print("Testing started.")

    # Testing case 1: Simple greeting
    print("Testing case [1/1] started.")
    english_sentence = 'Hello, how are you?'
    translation = translate_english_to_german(english_sentence)
    assert translation, f"Test case [1/1] failed: Expected a non-empty string, got '{translation}' instead."
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_english_to_german()