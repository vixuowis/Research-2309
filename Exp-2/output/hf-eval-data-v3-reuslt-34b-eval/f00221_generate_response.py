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

    # Load model and tokenizer
    dialogpt_model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
    dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

    # encode the new user input, add the eos_token and return a tensor in PyTorch
    new_user_input_ids = dialogpt_tokenizer.encode(user_input + dialogpt_tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history, new_user_input_ids], dim=-1) if chat_history is not None else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    dialogpt_output = dialogpt_model.generate(bot_input_ids, max_length=1000, pad_token_id=dialogpt_tokenizer.eos_token_id)
    
    # pretty print last ouput tokens from bot
    response = dialogpt_tokenizer.decode(dialogpt_output[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response.strip(), dialogpt_output

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