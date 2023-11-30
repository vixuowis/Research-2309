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
    model = AutoModelForSeq2SeqLM.from_pretrained('tuner007/pegasus_paraphrase')
    tokenizer = AutoTokenizer.from_pretrained('tuner007/pegasus_paraphrase')

    input_text = situation_narrative + ' ' + role_instruction + ' ' + '<newline>'.join(conversation_history)
    
    # Max model length: 512 tokens. Divide the input into 512-token segments and feed into the model as a batch.
    if len(tokenizer.encode(input_text)) <= 512:
        outputs = model.generate(input_text, do_sample=True, temperature=0.7)
    else:
        input_ids = tokenizer.batch_encode_plus([input_text], return_tensors='pt')['input_ids']
        
        n_chunks = torch.ceil(torch.tensor(len(input_ids[0]) / 512)).to(torch.int32)
        split_lengths = torch.ceil(torch.tensor(len(input_ids[0])) / n_chunks).to(torch.int32).repeat(n_chunks)
        
        # Ensure last chunk is less than 512 tokens in length (otherwise will throw error)
        split_lengths[-1] = len(input_ids[0]) - sum(split_lengths[:-1])
        
        chunks = torch.tensor([torch.cat([input_ids[0][i:i+l] for i in range(0, len(input_ids[0]), l)], dim=0).unsqueeze(0) \
                                  for l in split_lengths])[:n_chunks-1]
        
        outputs = model.generate(chunks, pad_token_id=50256, do_sample=True, temperature=0.7)    
            
    conversation = tokenizer.batch_decode(outputs, skip_special

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