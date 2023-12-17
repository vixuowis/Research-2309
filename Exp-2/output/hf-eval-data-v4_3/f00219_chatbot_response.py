# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

# function_code --------------------

def chatbot_response(message: str) -> str:
    """Generate a response to a given message using the BlenderBot model.

    Args:
        message (str): The message from the user to which the chatbot should respond.

    Returns:
        str: The chatbot's response to the user's message.

    Raises:
        ValueError: If the input message is empty or None.
    """
    if not message:
        raise ValueError('Input message is empty or None.')

    model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill')
    tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')
    inputs = tokenizer(message, return_tensors='pt')
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# test_function_code --------------------

def test_chatbot_response():
    print('Testing started.')

    # Test case 1: Normal input
    print('Testing case [1/1] started.')
    message = 'How can I cancel my subscription?'
    response = chatbot_response(message)
    assert response, f'Test case [1/1] failed: No response generated for valid input.'

    print('Testing finished.')


# call_test_function_line --------------------

test_chatbot_response()