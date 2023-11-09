# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# function_code --------------------

def generate_dialogue(user_input: str) -> str:
    """
    Generate a dialogue response using DialoGPT-large model.

    Args:
        user_input (str): The user's input to which the chatbot should respond.

    Returns:
        str: The chatbot's response.
    """
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
    model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-large')

    encoded_input = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    generated_response = model.generate(encoded_input, max_length=100, pad_token_id=tokenizer.eos_token_id)
    decoded_response = tokenizer.decode(generated_response[:, encoded_input.shape[-1]:][0], skip_special_tokens=True)

    return decoded_response

# test_function_code --------------------

def test_generate_dialogue():
    """
    Test the generate_dialogue function.
    """
    user_input = 'How do I search for scientific papers?'
    response = generate_dialogue(user_input)
    assert isinstance(response, str), 'The response should be a string.'
    assert response, 'The response should not be an empty string.'

# call_test_function_code --------------------

test_generate_dialogue()