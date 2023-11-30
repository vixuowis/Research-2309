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

    if not isinstance(situation_narrative, str) or \
            not isinstance(role_instruction, str) or \
            not isinstance(conversation_history, list):
        raise ValueError('Invalid input type.')
    
    # Set up model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained('/work/qfw1729-mcd1830/checkpoints/conversational-bart')
    model = AutoModelForSeq2SeqLM.from_pretrained('/work/qfw1729-mcd1830/checkpoints/conversational-bart').to(device)

    # Tokenize input and add EOS token to situation narrative, role instruction, and conversation history
    input_ids = tokenizer.batch_encode_plus([situation_narrative + ' ' + \
                    role_instruction + ' ' + ' '.join(conversation_history)])['input_ids'][0]  # List of integers
    
    # Set up decoding parameters and generate conversation using model.generate()
    eos_token_id = tokenizer.eos_token_id
    max_length = 40 + len(conversation_history)
    repetition_penalty = 1.3

    output_ids = model.generate(torch.tensor([input_ids]).to(device), \
                                max_length=max_length, \
                                eos_token_id=eos_token_id, \
                                repetition_penalty=repetition_penalty)
    conversation = tokenizer.decode(output_ids[0])  # str
    
    # Remove EOS token and return result
    conversation = conversation[:conversation.rfind('<EOS>')]
    if '<BOS>' in conversation:
        conversation = conversation[conversation.find('<BOS>'):][5:]
        
    return conversation

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