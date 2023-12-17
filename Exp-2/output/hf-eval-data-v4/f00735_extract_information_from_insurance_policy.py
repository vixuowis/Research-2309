# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_information_from_insurance_policy(image_path):
    # Initialize the document question-answering pipeline using Donut model
    doc_vqa = pipeline('document-question-answering', model='jinhybr/OCR-DocVQA-Donut')

    # Define a list of questions to extract relevant information from the insurance policy
    questions = [
        'What is the policy number?',
        'What is the coverage amount?',
        'Who is the beneficiary?',
        'What is the term period?'
    ]

    # Extract information from the insurance policy document image
    answers = {}
    for question in questions:
        result = doc_vqa(image_path=image_path, question=question)
        answers[question] = result['answer']
    return answers

# test_function_code --------------------

def test_extract_information_from_insurance_policy():
    print("Testing started.")
    sample_image_path = 'path/to/sample/insurance_document.jpg'

    # Run the extraction function on a sample document
    answers = extract_information_from_insurance_policy(sample_image_path)

    # Perform checks on the extracted data
    assert 'policy number' in answers['What is the policy number?'], "Test failed: Missing policy number."
    assert 'amount' in answers['What is the coverage amount?'], "Test failed: Missing coverage amount."
    assert 'beneficiary' in answers['Who is the beneficiary?'], "Test failed: Missing beneficiary name."
    assert 'term period' in answers['What is the term period?'], "Test failed: Missing term period."

    print("Testing finished successfully.")

# Run the test function
test_extract_information_from_insurance_policy()