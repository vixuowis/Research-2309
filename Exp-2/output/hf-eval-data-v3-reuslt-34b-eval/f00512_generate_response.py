# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

# function_code --------------------

def generate_response(input_text):
    """
    Generate a response to the input text using a pre-trained model.

    Args:
        input_text (str): The input text to which the model should respond.

    Returns:
        list: A list of generated responses.
    """
    
    # load tokenizer and model
    print('Loading tokenizer and model...')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    model = AutoModelWithLMHead.from_pretrained('../models/dialoGPT/output').cuda()

    # get response using model 
    print('Getting response...')
    chat_history_ids = []
    input_text += tokenizer.eos_token
    for _ in range(3):
        new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt').cuda()
        
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if len(chat_history_ids) > 0 else new_user_input_ids
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        
        chat_history = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        response = chat_history.split('\n')
        if response[-1] == '':
            response = response[:-1]
            
    return response

# test_function_code --------------------

def test_generate_response():
    """
    Test the generate_response function.
    """
    test_cases = [
        '@@ПЕРВЫЙ@@ привет @@ВТОРОЙ@@ привет @@ПЕРВЫЙ@@ как дела? @@ВТОРОЙ@@',
        '@@ПЕРВЫЙ@@ добрый день @@ВТОРОЙ@@ добрый день @@ПЕРВЫЙ@@ как погода? @@ВТОРОЙ@@',
        '@@ПЕРВЫЙ@@ здравствуйте @@ВТОРОЙ@@ здравствуйте @@ПЕРВЫЙ@@ что нового? @@ВТОРОЙ@@'
    ]
    for test_case in test_cases:
        result = generate_response(test_case)
        assert isinstance(result, list), 'The result should be a list.'
        assert len(result) > 0, 'The list should not be empty.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_response()