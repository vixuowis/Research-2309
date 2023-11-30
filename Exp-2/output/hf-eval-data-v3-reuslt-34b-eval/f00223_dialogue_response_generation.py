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
    
    # set up model name and tokenizer
    model_name = 'microsoft/DialoGPT-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True)
    
    # encode user input and generate chatbot output
    user_input_ids = tokenizer.encode(user_input+tokenizer.eos_token, \
                                      return_tensors='pt')
    bot_input_ids = torch.cat([user_input_ids, torch.zeros((1, 1), dtype=torch.int64)], dim=1)
    
    # set number of conversation steps
    for step in range(steps):
        outputs = model(bot_input_ids, labels=bot_input_ids)
        
        # get output tensor from the last step
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)[0].item() 
        
        bot_input_ids = torch.cat([bot_input_ids, torch.tensor([[next_token]])], \
                                  dim=1)
    
    # get output text and remove EOS token
    chatbot_response = tokenizer.decode(bot_input_ids[:, user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return chatbot_response

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