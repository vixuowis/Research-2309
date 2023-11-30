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

    if not user_input.strip(): # empty input
        return '' 
    
    inputs = tokenizer([user_input], return_tensors='pt')
        
    if not isinstance(chat_history, torch.Tensor) and chat_history is not None:
        raise ValueError('`chat_history` should be a `torch.Tensor`.')  # noqa
        
    bot_input = tokenizer([BOT], return_tensors='pt', padding=True, add_special_tokens=False)
    
    if isinstance(chat_history, torch.Tensor):
        inputs = torch.cat((chat_history, inputs['input_ids']), dim=-1)
        
    outputs = model.generate(inputs['input_ids'], max_length=CHAT_HISTORY_LENGTH+len(bot_input[0]) + 5, do_sample=True, top_k=70)
    response = tokenizer.decode(outputs[0], skip_special_tokens=False).split('Ġ')[1:] # [1:] removes the bot name in the beginning of each generated message
        
    response = ''.join([r for r in response if len(r)>1 and (r[-1] != tokenizer.eos_token or r[-2] != tokenizer.eos_token)])
    
    return response, outputs[0][-CHAT_HISTORY_LENGTH:]

# function_export --------------------


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