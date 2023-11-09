# function_import --------------------

from transformers import AutoTokenizer, AutoModelForCausalLM

# function_code --------------------

def generate_code(text):
    """
    Generate executable code based on the input prompt using the pre-trained model 'Salesforce/codegen-350M-multi'.

    Args:
        text (str): The input prompt in the form of a string.

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
    test_text = 'Create a simple loading spinner for maintenance.'
    expected_output = '...'
    assert generate_code(test_text) == expected_output

# call_test_function_code --------------------

test_generate_code()