# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_conversation(situation_narrative, role_instruction, conversation_history):
    """
    Generate a conversation based on a situation narrative, role instruction, and conversation history.

    Args:
        situation_narrative (str): The situation narrative.
        role_instruction (str): The role instruction.
        conversation_history (list): The conversation history.

    Returns:
        str: The generated conversation.
    """
    
    # load the model and tokenizer
    print('Loading the model...')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/DialoGPT-large").to("cuda")
    
    # generate a conversation history with the situation narrative and role instruction
    conversation_history += ['Human: ' + situation_narrative, 
                              'Agent: ' + role_instruction]

    # tokenize and create the input sequence
    conversation_history = [f'{i}: {conversation}' for i, conversation in enumerate(conversation_history)]
    
    inputs = tokenizer('\n'.join(conversation_history + ['>>>']), 
                       return_tensors='pt').to("cuda")
    
    # generate the output sequence
    reply = ""
    while len(reply.split()) <= 3:
        outputs = model.generate(**inputs, max_length=30)
        
        # decode the output sequence and add to conversation history
        reply = tokenizer.batch_decode(outputs)[0]
        conversation_history += ['Human: ' + reply]
        
        # generate input sequence for next dialog
        inputs = tokenizer('\n'.join(conversation_history), 
                           return_tensors='pt').to("cuda")
    
    return conversation_history, reply

# test_function_code --------------------

def test_generate_conversation():
    """
    Test the generate_conversation function.
    """
    situation = "Cosmo had a really fun time participating in the EMNLP conference at Abu Dhabi."
    instruction = "You are Cosmo and you are talking to a friend."
    conversation = ["Hey, how was your trip to Abu Dhabi?"]
    response = generate_conversation(situation, instruction, conversation)
    assert isinstance(response, str)
    assert len(response) > 0
    print('All Tests Passed')


# call_test_function_code --------------------

test_generate_conversation()