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
    # Set the seed for reproducibility.
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-66b")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-66b").to(device)
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0).to(device)
    output = model.generate(
        input_ids, 
        max_length=max_length,
        do_sample=True,  
        top_k=50, 
    )
    
    generated_texts = tokenizer.batch_decode(output)
    
    return generated_texts

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