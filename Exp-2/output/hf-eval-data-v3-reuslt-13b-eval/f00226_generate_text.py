# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch

# function_code --------------------

def generate_text(prompt: str, num_return_sequences: int = 5, max_length: int = 10):
    """
    Generate text based on a given prompt using the pretrained model 'facebook/opt-66b'.

    Args:
        prompt (str): The initial text to start the generation from.
        num_return_sequences (int, optional): The number of different response sequences to generate. Defaults to 5.
        max_length (int, optional): The maximum length of each response. Defaults to 10.

    Returns:
        List[str]: A list of generated text sequences.
    """
    
    set_seed(0)
    
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-66b", return_dict=True, force_download=False)
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-66b', use_fast=False)
    
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids  # encode the prompt
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_sequences)
    
    response_list = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in output]  # decode the responses to text
    return response_list

# test_function_code --------------------

def test_generate_text():
    """
    Test the function generate_text.
    """
    responses = generate_text('Hello, I am conscious and', 5, 10)
    assert isinstance(responses, list), 'The return type should be a list.'
    assert len(responses) == 5, 'The length of the list should be equal to num_return_sequences.'
    for response in responses:
        assert isinstance(response, str), 'Each element in the list should be a string.'
        assert len(response.split()) <= 10, 'The length of each response should be less than or equal to max_length.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_text()