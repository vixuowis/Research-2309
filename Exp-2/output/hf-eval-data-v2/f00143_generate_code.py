# function_import --------------------

from transformers import AutoTokenizer, AutoModelForCausalLM

# function_code --------------------

def generate_code(description):
    """
    Generate a code snippet based on a natural language description.

    Args:
        description (str): The natural language description of the code to be generated.

    Returns:
        str: The generated code snippet.
    """
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-2B-multi')
    model = AutoModelForCausalLM.from_pretrained('Salesforce/codegen-2B-multi')
    input_ids = tokenizer(description, return_tensors='pt').input_ids
    generated_ids = model.generate(input_ids, max_length=128)
    generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_code

# test_function_code --------------------

def test_generate_code():
    """
    Test the generate_code function.

    This function does not return anything but raises an error if the generate_code function does not work correctly.
    """
    description = 'Write a Python function to calculate the factorial of a number.'
    expected_output = 'def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)'
    output = generate_code(description)
    assert 'factorial' in output, 'Test failed!'
    assert 'n' in output, 'Test failed!'
    assert 'return' in output, 'Test failed!'

# call_test_function_code --------------------

test_generate_code()