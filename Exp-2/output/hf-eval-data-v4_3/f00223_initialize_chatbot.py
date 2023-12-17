# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# function_code --------------------

def initialize_chatbot(model_name: str):
    """
    Initializes the chatbot by creating a tokenizer and loading a pretrained DialoGPT model.

    Args:
        model_name: A string containing the model identifier for a pretrained model.

    Returns:
        A tuple containing a tokenizer and the loaded model.

    Raises:
        ValueError: If `model_name` is an empty string.
    """
    if not model_name:
        raise ValueError("Model name cannot be an empty string.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# test_function_code --------------------

def test_initialize_chatbot():
    print("Testing started.")

    # Test case 1: Valid model name and assertions
    print("Testing case [1/3] started.")
    tokenizer, model = initialize_chatbot('microsoft/DialoGPT-small')
    assert isinstance(tokenizer, AutoTokenizer), "Test case [1/3] failed: tokenizer is not an instance of AutoTokenizer."
    assert isinstance(model, AutoModelForCausalLM), "Test case [1/3] failed: model is not an instance of AutoModelForCausalLM."

    # Test case 2: Empty model name and error handling
    print("Testing case [2/3] started.")
    try:
        initialize_chatbot('')
    except ValueError as e:
        assert str(e) == 'Model name cannot be an empty string.', "Test case [2/3] failed: ValueError not raised as expected."

    print("Testing finished.")

# call_test_function_line --------------------

test_initialize_chatbot()