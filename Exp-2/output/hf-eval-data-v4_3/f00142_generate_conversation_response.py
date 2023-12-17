# requirements_file --------------------

import subprocess

requirements = ["torch", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_conversation_response(situation_narrative, role_instruction, conversation_history):
    """
    Generate a response for a conversation based on a situation narrative and role instruction.

    Args:
        situation_narrative (str): Description of the situation.
        role_instruction (str): The role the AI plays in the conversation.
        conversation_history (List[str]): History of the conversation.

    Returns:
        str: The generated response to add to the conversation.

    Raises:
        ValueError: If conversation_history is not a list.
    """
    if not isinstance(conversation_history, list):
        raise ValueError('conversation_history argument must be a list')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('allenai/cosmo-xl')
    model = AutoModelForSeq2SeqLM.from_pretrained('allenai/cosmo-xl').to(device)
    input_text = set_input(situation_narrative, role_instruction, conversation_history)
    inputs = tokenizer([input_text], return_tensors='pt').to(device)
    outputs = model.generate(inputs['input_ids'], max_new_tokens=128, temperature=1.0, top_p=.95, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return response

def set_input(situation_narrative, role_instruction, conversation_history):
    input_text = ' <turn> '.join(conversation_history)
    if role_instruction != '':
        input_text = '{} <sep> {}'.format(role_instruction, input_text)
    if situation_narrative != '':
        input_text = '{} <sep> {}'.format(situation_narrative, input_text)
    return input_text

# test_function_code --------------------

def test_generate_conversation_response():
    print('Testing started.')
    
    # Test case 1: Standard conversation flow
    print('Testing case [1/3] started.')
    situation = 'Cosmo had a really fun time participating in the EMNLP conference at Abu Dhabi.'
    instruction = 'You are Cosmo and you are talking to a friend.'
    conversation_history = ['Hey, how was your trip to Abu Dhabi?']
    response = generate_conversation_response(situation, instruction, conversation_history)
    assert isinstance(response, str), f'Test case [1/3] failed: Expected a string response, got {type(response)}'

    # Test case 2: No role instruction provided
    print('Testing case [2/3] started.')
    conversation_history.append(response)
    response = generate_conversation_response(situation, '', conversation_history)
    assert isinstance(response, str), f'Test case [2/3] failed: Expected a string response, got {type(response)}'

    # Test case 3: Empty conversation history
    print('Testing case [3/3] started.')
    response = generate_conversation_response(situation, instruction, [])
    assert isinstance(response, str), f'Test case [3/3] failed: Expected a string response, got {type(response)}'
    
    print('Testing finished.')

# call_test_function_line --------------------

test_generate_conversation_response()