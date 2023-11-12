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
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-350M-multi')
    model = AutoModelForCausalLM.from_pretrained('Salesforce/codegen-350M-multi')
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    generated_ids = model.generate(input_ids, max_length=128)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

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