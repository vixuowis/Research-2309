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

    # Load the tokenizer and model from Hugging Face.
    tokenizer = AutoTokenizer.from_pretrained('yjernite/retro-future')
    model = AutoModelWithLMHead.from_pretrained('yjernite/retro-future')

    # Tokenize the input text and add the eos_token.
    input_ids = torch.tensor(tokenizer.encode(input_text + tokenizer.eos_token)).unsqueeze(0)  # Batch size 1
    
    # Set the model in evaluation mode.
    model.eval()
    
    # Generate a response (the PREDICT method). The top_k is set to 5, meaning that the model only needs to find
    # five answers which all have a high probability of being correct.
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=128, do_sample=True, top_p=0.95, top_k=5)
        
    # Get and print the generated responses from the model.
    response_list = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
    
    return response_list

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