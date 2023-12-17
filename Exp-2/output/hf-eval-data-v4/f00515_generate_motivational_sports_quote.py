# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_motivational_sports_quote(prompt='Motivational quote about sports:', max_length=50):
    """
    Generate a motivational quote related to sports using a pre-trained text-generation model.
    
    Parameters:
        prompt (str): The initial text to prompt the model with.
        max_length (int): The maximum length of the generated text.
    
    Returns:
        str: The generated motivational sports quote.
    """
    # Initialize the text-generation pipeline with the specified model
    text_generator = pipeline('text-generation', model='TehVenom/PPO_Pygway-V8p4_Dev-6b')
    # Generate text based on the prompt
    generated_text = text_generator(prompt, max_length=max_length)[0]['generated_text']
    return generated_text.strip()

# test_function_code --------------------

def test_generate_motivational_sports_quote():
    print("Testing generate_motivational_sports_quote function.")
    
    # Test case 1: Check if the function returns a string
    print("Testing case [1/1] started.")
    quote = generate_motivational_sports_quote()
    assert isinstance(quote, str), f"Test case [1/1] failed: Expected a string, got {type(quote).__name__} instead."
    print("Test case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_generate_motivational_sports_quote()