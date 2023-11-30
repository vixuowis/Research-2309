# function_import --------------------

from transformers import AutoTokenizer, AutoModelForCausalLM

# function_code --------------------

def generate_code(text):
    """
    Generate executable code based on the input prompt using Hugging Face Transformers.

    Args:
        text (str): The input prompt in English.

    Returns:
        str: The generated executable code.
    """

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-python-base")
    model = AutoModelForCausalLM.from_pretrained("microsoft/codebert-python-base")

    # Tokenize input prompt
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    # Generate code using the model
    generated_ids = model.generate(input_ids=input_ids, max_length=1000)[0]
    generated_code = tokenizer.decode(generated_ids)

    return generated_code

# test_function_code --------------------

def test_generate_code():
    """
    Test the function generate_code.
    """
    assert generate_code('Create a simple loading spinner for maintenance.') is not None
    assert generate_code('Create a function to add two numbers.') is not None
    assert generate_code('Create a function to calculate the factorial of a number.') is not None
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_code()