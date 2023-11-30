# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

# function_code --------------------

def generate_dialogue(input_text):
    """
    Generate a dialogue in Russian using a pretrained model.

    Args:
        input_text (str): The input text in Russian to generate a dialogue from.

    Returns:
        list: A list of generated dialogues.
    """
    
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/russian-gpt2-conversational")
    model = AutoModelWithLMHead.from_pretrained("DeepPavlov/russian-gpt2-conversational", return_dict=True)
    
    input_text += ':'
    tokens = tokenizer(input_text, return_tensors='pt')['input_ids'][:, -1:]
        
    chatbot_response = []
    
    for i in range(3): # generate three dialogues at most.
        outputs = model.generate(tokens, max_length=512, repetition_penalty=2.0, top_k=40, do_sample=True) 
        
        chatbot_response += [tokenizer.decode(x, skip_special_tokens=True).replace('<|endoftext|>', '') for x in outputs][:1] # the second argument returns only one dialogue.
                    
        if ':' not in chatbot_response[i]: break
        
        tokens = tokenizer(input_text + chatbot_response[-1], return_tensors='pt')['input_ids'][:, -1:] # update the input text for the next dialogue
    
    return chatbot_response

# test_function_code --------------------

def test_generate_dialogue():
    """
    Test the generate_dialogue function.
    """
    input_text = '@@ПЕРВЫЙ@@ привет @@ВТОРОЙ@@ привет @@ПЕРВЫЙ@@ как дела?'
    output = generate_dialogue(input_text)
    assert isinstance(output, list), 'Output should be a list.'
    assert len(output) > 0, 'Output list should not be empty.'
    assert all(isinstance(i, str) for i in output), 'All elements in the output list should be strings.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_dialogue()