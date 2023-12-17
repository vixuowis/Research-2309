# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer

# function_code --------------------

def extract_total_amount(question, context):
    """
    Extracts the total amount due from an invoice document given the question and context.

    Args:
        question (str): The question to be answered.
        context (str): The context which contains the answer to the question.

    Returns:
        str: The answer to the question extracted from the context.

    Raises:
        ValueError: If the model or tokenizer could not be loaded properly.
    """
    try:
        model = AutoModelForDocumentQuestionAnswering.from_pretrained('impira/layoutlm-invoices')
        tokenizer = AutoTokenizer.from_pretrained('impira/layoutlm-invoices')
    except Exception as e:
        raise ValueError('Failed to load model or tokenizer.') from e

    inputs = tokenizer(question, context, return_tensors='pt')
    outputs = model(**inputs)
    answer_start = outputs.start_logits.argmax().item()
    answer_end = outputs.end_logits.argmax().item()
    answer = tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end+1].tolist())
    return answer

# test_function_code --------------------

def test_extract_total_amount():
    print('Testing started.')

    # Test case 1: Invoice with total amount
    print('Testing case [1/1] started.')
    question = 'What is the total amount?'
    context = 'Invoice information for order ABC_123\nProduct: Widget A, Quantity: 10, Price: $5 each\nProduct: Widget B, Quantity: 5, Price: $3 each\nProduct: Widget C, Quantity: 15, Price: $2 each\nSubtotal: $75, Tax: $6.38, Total Amount Due: $81.38'
    expected_answer = '$81.38'
    answer = extract_total_amount(question, context)
    assert answer == expected_answer, f'Test case [1/1] failed: Expected {expected_answer} but got {answer}'
    print('Testing finished.')

# call_test_function_line --------------------

test_extract_total_amount()