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
    # Load the tokenizer and model from huggingface hub
    tokenizer = AutoTokenizer.from_pretrained('planb18/GPT2LMHeadModel')
    model = AutoModelForCausalLM.from_pretrained('planb18/GPT2LMHeadModel')

    # Generate code
    input = "'''#description: " + description + "'''\n"
    output = tokenizer.decode(model.generate(tokenizer.encode(input), max_length=300)[0])[len(input):]
    return output

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