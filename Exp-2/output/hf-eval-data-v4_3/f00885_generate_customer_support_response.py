# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_customer_support_response(instruction, knowledge, dialog):
    """
    Generate a response to a customer's complaint about late delivery using GODEL model.

    Args:
        instruction (str): The instruction to generate a response for.
        knowledge (str): External knowledge relevant to the complaint.
        dialog (list): A list containing the dialog history.

    Returns:
        str: Suggested customer support response.

    Raises:
        ValueError: If any of the arguments are not of the expected type.
    """
    if not isinstance(instruction, str) or not isinstance(knowledge, str) or not isinstance(dialog, list):
        raise ValueError('Arguments must be of type str and list for dialog.')
    knowledge_prefix = '[KNOWLEDGE] ' + knowledge if knowledge else ''
    dialog_str = ' EOS '.join(dialog)
    query = f'{instruction} [CONTEXT] {dialog_str} {knowledge_prefix}'
    tokenizer = AutoTokenizer.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
    model = AutoModelForSeq2SeqLM.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
    input_ids = tokenizer.encode(query, return_tensors='pt')
    output_ids = model.generate(input_ids, max_length=128, min_length=20, num_beams=5)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# test_function_code --------------------

def test_generate_customer_support_response():
    print('Testing started.')
    # Test case 1: Check response type
    print('Testing case [1/2] started.')
    instruction = 'How can I respond to a customer complaint about late delivery?'
    knowledge = 'The courier had external delays due to bad winter weather.'
    dialog = ['Customer: My package is late. What\'s going on?', 'Support: I apologize for the inconvenience. I\'ll check what\'s happening with the package and get back to you.']
    response = generate_customer_support_response(instruction, knowledge, dialog)
    assert isinstance(response, str), f'Test case [1/2] failed: The response must be a string.'

    # Test case 2: Check for ValueError when invalid arguments are passed
    print('Testing case [2/2] started.')
    invalid_args = (123, ['not', 'valid'], 'Incorrect dialog type')
    try:
        generate_customer_support_response(*invalid_args)
        assert False, 'Test case [2/2] failed: Expected ValueError for invalid arguments.'
    except ValueError:
        pass

    print('Testing finished.')

# call_test_function_line --------------------

test_generate_customer_support_response()