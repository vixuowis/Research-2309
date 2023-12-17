# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# function_code --------------------

def chat_with_blenderbot(input_message: str) -> str:
    """Generate a conversational response from BlenderBot.

    Args:
        input_message (str): The input message to the chatbot.
    
    Returns:
        str: The chatbot's conversational response.

    Raises:
        ValueError: If the input message is empty.
    """
    if not input_message:
        raise ValueError('The input message cannot be empty.')
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/blenderbot-1B-distill')
    tokenizer = AutoTokenizer.from_pretrained('facebook/blenderbot-1B-distill')
    inputs = tokenizer(input_message, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'])
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

# test_function_code --------------------

def test_chat_with_blenderbot():
    print("Testing started.")

    # Test case 1: Normal input
    print("Testing case [1/2] started.")
    response = chat_with_blenderbot('Hello, how are you?')
    assert isinstance(response, str) and len(response) > 0, f"Test case [1/2] failed: Response should be a non-empty string, got {response}"

    # Test case 2: Empty input
    print("Testing case [2/2] started.")
    try:
        response = chat_with_blenderbot('')
        assert False, "Test case [2/2] failed: ValueError not raised for empty input."
    except ValueError:
        pass  # Expected

    print("Testing finished.")

# call_test_function_line --------------------

test_chat_with_blenderbot()