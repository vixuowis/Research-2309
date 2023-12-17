# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def chat_with_blenderbot(input_text):
    """
    Generate a response from BlenderBot based on the input text provided by user.

    Args:
        input_text (str): The input text to which the bot should respond.

    Returns:
        str: The response generated by BlenderBot.
    """
    chatbot = pipeline('conversational', model='hyunwoongko/blenderbot-9B')
    response = chatbot(input_text)
    return response

# test_function_code --------------------

def test_chat_with_blenderbot():
    print("Testing started.")
    
    # Test case 1: Greeting
    print("Testing case [1/3] started.")
    response = chat_with_blenderbot("Hello!")
    assert isinstance(response, str), f"Test case [1/3] failed: Expected string response, got {type(response)}"

    # Test case 2: Question about music
    print("Testing case [2/3] started.")
    response = chat_with_blenderbot("What is your favorite type of music?")
    assert isinstance(response, str), f"Test case [2/3] failed: Expected string response, got {type(response)}"

    # Test case 3: Goodbye
    print("Testing case [3/3] started.")
    response = chat_with_blenderbot("Goodbye!")
    assert isinstance(response, str), f"Test case [3/3] failed: Expected string response, got {type(response)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_chat_with_blenderbot()