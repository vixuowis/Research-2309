# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_invoice_details(image_path):
    """
    Extracts details such as total amount, date of invoice, and name of the service provider from an invoice.

    Args:
        image_path (str): The path to the image file of the invoice.

    Returns:
        dict: A dictionary containing the extracted information.

    Raises:
        Exception: If there is an error in processing the invoice.
    """
    # Initialize the document-question-answering pipeline with the OCR-DocVQA-Donut model
    doc_vqa = pipeline('document-question-answering', model='jinhybr/OCR-DocVQA-Donut')

    # Define the questions to extract the required information
    questions = [
        'What is the total amount?',
        'What is the date of the invoice?',
        'What is the name of the service provider?'
    ]

    # Extract information for each question
    answers = {q: doc_vqa(image_path=image_path, question=q)['answer'] for q in questions}

    return answers

# test_function_code --------------------

def test_extract_invoice_details():
    print("Testing started.")

    # This is a placeholder image path and expected results
    image_path = 'path/to/invoice_image.jpg'
    expected_results = {
        'What is the total amount?': '$100.00',
        'What is the date of the invoice?': '2021-08-01',
        'What is the name of the service provider?': 'XYZ Corp'
    }

    # Test case 1: Check if the function can extract the total amount
    print("Testing case [1/3] started.")
    extraction = test_extract_invoice_details(image_path)
    assert extraction['What is the total amount?'] == expected_results['What is the total amount?'], f"Test case [1/3] failed: Expected {expected_results['What is the total amount?']}, got {extraction['What is the total amount?']}"

    # Test case 2: Check if the function can extract the date of the invoice
    print("Testing case [2/3] started.")
    assert extraction['What is the date of the invoice?'] == expected_results['What is the date of the invoice?'], f"Test case [2/3] failed: Expected {expected_results['What is the date of the invoice?']}, got {extraction['What is the date of the invoice?']}"

    # Test case 3: Check if the function can extract the name of the service provider
    print("Testing case [3/3] started.")
    assert extraction['What is the name of the service provider?'] == expected_results['What is the name of the service provider?'], f"Test case [3/3] failed: Expected {expected_results['What is the name of the service provider?']}, got {extraction['What is the name of the service provider?']}"
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_invoice_details()