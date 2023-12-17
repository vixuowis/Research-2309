# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForCausalLM

# function_code --------------------

def generate_code_snippet(description: str) -> str:
    """
    Generate a code snippet based on a natural language description.

    Args:
        description (str): A natural language description of the desired code.

    Returns:
        str: The generated code snippet.

    Raises:
        ValueError: If the description is empty or not provided.
    """
    if not description:
        raise ValueError('Description is required')

    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-2B-multi')
    model = AutoModelForCausalLM.from_pretrained('Salesforce/codegen-2B-multi')
    input_ids = tokenizer(description, return_tensors='pt').input_ids
    generated_ids = model.generate(input_ids, max_length=128)
    code_snippet = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return code_snippet

# test_function_code --------------------

def test_generate_code_snippet():
    print("Testing started.")

    # Test case 1: Valid description provided
    print("Testing case [1/2] started.")
    description = 'Write a Python function to calculate the factorial of a number.'
    generated_snippet = generate_code_snippet(description)
    assert 'def' in generated_snippet and 'factorial' in generated_snippet, f"Test case [1/2] failed: Generated snippet does not contain expected elements."

    # Test case 2: Empty description
    print("Testing case [2/2] started.")
    empty_description = ''
    try:
        generate_code_snippet(empty_description)
        assert False, "Test case [2/2] failed: Empty description should raise ValueError."
    except ValueError as e:
        assert str(e) == 'Description is required', f"Test case [2/2] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_code_snippet()