# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_conversation(situation_narrative, role_instruction, conversation_history):
    """
    This function generates a conversation based on a given situation narrative, role instruction, and conversation history.

    Args:
        situation_narrative (str): The situation narrative.
        role_instruction (str): The role instruction.
        conversation_history (list): The conversation history.

    Returns:
        str: The generated conversation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('allenai/cosmo-xl')
    model = AutoModelForSeq2SeqLM.from_pretrained('allenai/cosmo-xl').to(device)

    def set_input(situation_narrative, role_instruction, conversation_history):
        input_text = " <turn> ".join(conversation_history)
        if role_instruction != "":
            input_text = "{} <sep> {}".format(role_instruction, input_text)
        if situation_narrative != "":
            input_text = "{} <sep> {}".format(situation_narrative, input_text)
        return input_text

    input_text = set_input(situation_narrative, role_instruction, conversation_history)
    inputs = tokenizer([input_text], return_tensors='pt').to(device)
    outputs = model.generate(inputs['input_ids'], max_new_tokens=128, temperature=1.0, top_p=.95, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return response

# test_function_code --------------------

def test_generate_conversation():
    """
    This function tests the generate_conversation function.
    """
    situation = "Cosmo had a really fun time participating in the EMNLP conference at Abu Dhabi."
    instruction = "You are Cosmo and you are talking to a friend."
    conversation = ["Hey, how was your trip to Abu Dhabi?"]
    response = generate_conversation(situation, instruction, conversation)
    assert isinstance(response, str)

# call_test_function_code --------------------

test_generate_conversation()