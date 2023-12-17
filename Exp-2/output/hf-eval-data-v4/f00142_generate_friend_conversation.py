# requirements_file --------------------

!pip install -U torch transformers

# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_friend_conversation(situation, instruction, conversation):
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

    def generate(situation_narrative, role_instruction, conversation_history):
        input_text = set_input(situation_narrative, role_instruction, conversation_history)
        inputs = tokenizer([input_text], return_tensors='pt').to(device)
        outputs = model.generate(inputs['input_ids'], max_new_tokens=128, temperature=1.0, top_p=.95, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return response

    response = generate(situation, instruction, conversation)
    return response

# test_function_code --------------------

def test_generate_friend_conversation():
    print("Testing started.")

    # Test case 1: Simple conversation starter
    print("Testing case [1/3] started.")
    situation = "Cosmo had a really fun time participating in the EMNLP conference at Abu Dhabi."
    instruction = "You are Cosmo and you are talking to a friend."
    conversation = ["Hey, how was your trip to Abu Dhabi?"]
    response = generate_friend_conversation(situation, instruction, conversation)
    assert type(response) == str and len(response) > 0, f"Test case [1/3] failed: The response should be a non-empty string."
    print("Testing case [1/3] passed.")

    # Test case 2: Continuing an ongoing conversation
    print("Testing case [2/3] started.")
    conversation.append(response) # Append the previous response
    new_response = generate_friend_conversation(situation, instruction, conversation)
    assert type(new_response) == str and new_response != response, f"Test case [2/3] failed: The new response should be different from the previous one."
    print("Testing case [2/3] passed.")

    # Test case 3: Checking for role adherence
    print("Testing case [3/3] started.")
    instruction = "You are now a conference organizer talking to Cosmo."
    organizer_response = generate_friend_conversation(situation, instruction, conversation)
    assert type(organizer_response) == str, f"Test case [3/3] failed: The response should be a string."
    print("Testing case [3/3] passed.")

    print("Testing finished.")

# Run the test function
test_generate_friend_conversation()