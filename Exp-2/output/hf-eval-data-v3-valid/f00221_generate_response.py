# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# function_code --------------------

def generate_response(user_input: str, chat_history: torch.Tensor = None):
    """
    Generate a response for the given user input using DialoGPT-medium model.

    Args:
        user_input (str): The input message from user.
        chat_history (torch.Tensor, optional): The chat history. Defaults to None.

    Returns:
        Tuple[str, torch.Tensor]: The generated response and the updated chat history.
    """
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')

    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    chat_history = torch.cat([chat_history, input_ids], dim=-1) if chat_history is not None else input_ids
    outputs = model.generate(chat_history, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, outputs

# test_function_code --------------------

def test_generate_response():
    user_input = 'Hello, how are you?'
    response, chat_history = generate_response(user_input)
    assert isinstance(response, str)
    assert isinstance(chat_history, torch.Tensor)
    user_input = 'What is your name?'
    response, chat_history = generate_response(user_input, chat_history)
    assert isinstance(response, str)
    assert isinstance(chat_history, torch.Tensor)
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_generate_response())