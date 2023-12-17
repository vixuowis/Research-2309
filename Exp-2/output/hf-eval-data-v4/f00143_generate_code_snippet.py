# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForCausalLM

# function_code --------------------

def generate_code_snippet(description: str) -> str:
    """
    Generate a code snippet based on a given natural language description.

    Parameters:
    description (str): A natural language description of the code to generate.

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

def test_generate_code_snippet():
    print("Testing generate_code_snippet function.")

    # Test case 1: Generate code for 'Hello World' program
    description = 'Write a Python function to print Hello World.'
    generated_code = generate_code_snippet(description)
    assert 'def' in generated_code and 'print' in generated_code and 'Hello World' in generated_code, f"Test case failed: {generated_code}"

    # Add more test cases if necessary

    print("Testing finished successfully.")
