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

     # load model and tokenizer --------------------

    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
    
    # prepare input to the model --------------------
    
    user_input = user_input + tokenizer.eos_token
    chat_ids = [int(x) for x in tokenizer.encode(user_input)]
    bot_id = torch.tensor([0]).unsqueeze(0).long()
    
    with torch.no_grad():
        outputs, _ = model(torch.tensor([chat_ids]), bot_id) # chat ids, bot id
        
    response = ''
    for i in range(steps):
        next_logits = outputs[:,-1,:]
        next_tokens = torch.argmax(next_logits, dim=-1).unsqueeze(0)
        chat_ids = torch.cat([chat_ids, next_tokens], dim=-1)
        
        response += tokenizer.decode(chat_ids[:, -1].tolist(), clean_up_tokenization_spaces=True)
    
    return response

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