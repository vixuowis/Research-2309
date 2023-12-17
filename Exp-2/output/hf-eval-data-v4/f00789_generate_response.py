# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# function_code --------------------

def generate_response(user_input):
    """
    This function takes a string as input, which represents the user's message to the chatbot.
    It then encodes the message, uses DialoGPT to generate a response, and decodes the response.
    
    Parameters:
    user_input (str): User's input message to chatbot.
    
    Returns:
    str: The chatbot's generated response.
    """
    # Instantiate tokenizer and model objects using pre-trained "microsoft/DialoGPT-large"
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
    model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-large')
    
    # Tokenize and encode the user input
    encoded_input = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # Generate a response using DialoGPT model
    generated_response = model.generate(encoded_input, max_length=100, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the generated response back to a string
    decoded_response = tokenizer.decode(generated_response[:, encoded_input.shape[-1]:][0], skip_special_tokens=True)
    return decoded_response

# test_function_code --------------------

def test_generate_response():
    print("Testing started.")
    
    # Test case 1: Using a common greeting
    print("Testing case [1/3] started.")
    user_input = "Hello, how are you?"
    response = generate_response(user_input)
    assert type(response) == str, f"Test case [1/3] failed: Response should be a string, but got {type(response)}"
    
    # Test case 2: Abstract question
    print("Testing case [2/3] started.")
    user_input = "What is the meaning of life?"
    response = generate_response(user_input)
    assert type(response) == str, f"Test case [2/3] failed: Response should be a string, but got {type(response)}"
    
    # Test case 3: Technical question
    print("Testing case [3/3] started.")
    user_input = "Can you recommend a good statistical method for data analysis?"
    response = generate_response(user_input)
    assert type(response) == str, f"Test case [3/3] failed: Response should be a string, but got {type(response)}"
    
    print("All tests passed.")
    
# Run the test function
test_generate_response()