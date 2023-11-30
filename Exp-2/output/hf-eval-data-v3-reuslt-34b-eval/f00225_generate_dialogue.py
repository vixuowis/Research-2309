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

    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-base-cased-conversational")
    model = AutoModelWithLMHead.from_pretrained("cointegrated/rubert-base-cased-conversational")

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids=input_ids)
    
    dialogue = [tokenizer.decode(output[0])]

    return dialogue

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