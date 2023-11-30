# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# function_code --------------------

def dialogue_response_generation(user_input: str, steps: int = 5):
    '''
    Generate dialogue response using DialoGPT model.

    Args:
        user_input (str): The user input to the chatbot.
        steps (int, optional): The number of conversation steps. Defaults to 5.

    Returns:
        str: The generated dialogue response.
    '''

    # Initialize device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to(device)

    # Set the start of sequence.
    chat_history_ids = tokenizer.encode('Robot:', add_special_tokens=False) + \
        tokenizer.encode(user_input, add_special_tokens=False)

    for step in range(steps):
        # encode the new user input, add the eos_token and return a tensor
        new_user_input_ids = tokenizer.encode(f'Human:', add_special_tokens=False) + \
            tokenizer.encode(f'{step+1}. ', add_special_tokens=False) + \
            tokenizer.encode('Hello! How can I help you?', add_special_tokens=False) + [tokenizer.eos_token_id]

        input_ids = chat_history_ids + new_user_input_ids
        input_ids = torch.tensor([input_ids]).to(device)
        
        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(
            input_ids, max_length=2000, pad_token_id=tokenizer.eos_token_id)
        
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


# test_function_code --------------------

def test_dialogue_response_generation():
    '''
    Test the dialogue_response_generation function.
    '''
    response = dialogue_response_generation('Hello, how are you?', 1)
    assert isinstance(response, str), 'The response should be a string.'
    assert 'DialoGPT:' in response, 'The response should start with DialoGPT:.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_dialogue_response_generation()