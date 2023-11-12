# function_import --------------------

from transformers import AutoTokenizer, AutoModelForCausalLM

# function_code --------------------

def generate_code(description: str) -> str:
    '''
    Generate code snippets based on natural language descriptions.

    Args:
        description (str): The natural language description.

    Returns:
        str: The generated code snippet.
    '''
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-2B-multi')
    model = AutoModelForCausalLM.from_pretrained('Salesforce/codegen-2B-multi')
    input_ids = tokenizer(description, return_tensors='pt').input_ids
    generated_ids = model.generate(input_ids, max_length=128)
    generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_code

# test_function_code --------------------

def test_generate_code():
    '''
    Test the generate_code function.
    '''
    description1 = 'Write a Python function to calculate the factorial of a number.'
    description2 = 'Write a Python function to sort a list of numbers in ascending order.'
    description3 = 'Write a Python function to reverse a string.'
    assert isinstance(generate_code(description1), str)
    assert isinstance(generate_code(description2), str)
    assert isinstance(generate_code(description3), str)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_code()