# requirements_file --------------------

import subprocess

requirements = ["torch", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

# function_code --------------------

def generate_russian_dialogue(input_text):
    """
    Generates a dialogue in Russian based on the given input text.

    Args:
        input_text (str): The initial text of the conversation.

    Returns:
        List[str]: Generated dialogue responses.

    Raises:
        ValueError: If the input_text is empty or None.
    """
    if not input_text:
        raise ValueError('input_text is required and cannot be empty.')

    tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
    model = AutoModelWithLMHead.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
    inputs = tokenizer(input_text, return_tensors='pt')
    generated_token_ids = model.generate(
        **inputs,
        max_length=50,
        num_beams=5,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )
    return [tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in generated_token_ids]

# test_function_code --------------------

def test_generate_russian_dialogue():
    print('Testing started.')
    # This function does not require loading a dataset as it is testing the dialogue generation.

    # Test case 1: Valid input text
    print('Testing case [1/3] started.')
    responses = generate_russian_dialogue('@@ПЕРВЫЙ@@ привет @@ВТОРОЙ@@ привет @@ПЕРВЫЙ@@ как дела?')
    assert responses, 'Test case [1/3] failed: No response generated.'

    # Test case 2: Empty input text
    print('Testing case [2/3] started.')
    try:
        generate_russian_dialogue('')
    except ValueError as e:
        assert str(e) == 'input_text is required and cannot be empty.', f'Test case [2/3] failed: {str(e)}'

    # Test case 3: None as input text
    print('Testing case [3/3] started.')
    try:
        generate_russian_dialogue(None)
    except ValueError as e:
        assert str(e) == 'input_text is required and cannot be empty.', f'Test case [3/3] failed: {str(e)}'

    print('Testing finished.')

# call_test_function_line --------------------

test_generate_russian_dialogue()