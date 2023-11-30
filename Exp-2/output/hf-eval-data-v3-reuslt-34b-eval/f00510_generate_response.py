# function_import --------------------

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# function_code --------------------

def generate_response(user_input, model_name='microsoft/DialoGPT-small', max_length=500, no_repeat_ngram_size=3, do_sample=True, top_k=100, top_p=0.7, temperature=0.8):
    """
    Generate a response from the AI model based on the user input.

    Args:
        user_input (str): The input from the user.
        model_name (str, optional): The name of the pre-trained model. Defaults to 'microsoft/DialoGPT-small'.
        max_length (int, optional): The maximum length of the generated response. Defaults to 500.
        no_repeat_ngram_size (int, optional): The size of the no repeat n-gram. Defaults to 3.
        do_sample (bool, optional): Whether to sample the response. Defaults to True.
        top_k (int, optional): The number of top k predictions to consider. Defaults to 100.
        top_p (float, optional): The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Defaults to 0.7.
        temperature (float, optional): The value used to module the next token probabilities. Defaults to 0.8.

    Returns:
        str: The generated response from the AI model.
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    
    # Process the user input
    history = [user_input]
    inputs = tokenizer(history, return_tensors='pt', padding=True).to(device)
    
    output = model.generate(
            **inputs, 
            max_length = max_length,
            no_repeat_ngram_size = no_repeat_ngram_size,
            do_sample = do_sample,
            top_k = top_k,
            top_p = top_p,
            temperature = temperature)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True).replace("&", 'and')
    return response

# test_function_code --------------------

def test_generate_response():
    """
    Test the generate_response function.
    """
    user_input = 'Hello, how are you?'
    response = generate_response(user_input)
    assert isinstance(response, str)
    assert len(response) > 0

    user_input = 'What is your name?'
    response = generate_response(user_input)
    assert isinstance(response, str)
    assert len(response) > 0

    user_input = 'Tell me a joke.'
    response = generate_response(user_input)
    assert isinstance(response, str)
    assert len(response) > 0

    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_response()