# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def translate_question_to_german(question: str) -> str:
    """
    Translates an English question to German, specifically asking about the location of parks in Munich.

    Args:
        question (str): The question in English to be translated.

    Returns:
        str: The translated question in German.

    Raises:
        ValueError: If the input question is empty.

    """
    if not question:
        raise ValueError('Input question is empty')

    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
    input_text = f"translate English to German: {question}"
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    outputs = model.generate(input_ids)
    translated_question = tokenizer.decode(outputs[0])
    return translated_question

# test_function_code --------------------

def test_translate_question_to_german():
    print('Testing started.')
    # Test Case 1: Check non-empty input
    print('Testing case [1/3] started.')
    question = 'Where are the parks in Munich?'
    result = translate_question_to_german(question)
    assert result, f'Test case [1/3] failed: Expected non-empty output, got {result}'

    # Test Case 2: Check input is correctly translated (this will be a placeholder as we cannot check exact translation)
    print('Testing case [2/3] started.')
    expected_start = 'Wo'
    assert result.startswith(expected_start), f'Test case [2/3] failed: Expected output to start with {expected_start}, got {result}'
    
    # Test Case 3: Check empty input raises ValueError
    print('Testing case [3/3] started.')
    try:
        translate_question_to_german('')
        assert False, 'Test case [3/3] failed: ValueError not raised for empty input'
    except ValueError:
        pass  # Expected
    print('Testing finished.')

# call_test_function_line --------------------

test_translate_question_to_german()