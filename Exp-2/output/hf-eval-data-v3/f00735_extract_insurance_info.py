# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_insurance_info(image_path: str) -> dict:
    """
    Extracts relevant information from an insurance policy document image using the Hugging Face Transformers pipeline.

    Args:
        image_path (str): The path to the insurance policy document image.

    Returns:
        dict: A dictionary where the keys are the questions asked and the values are the answers.
    """
    doc_vqa = pipeline('document-question-answering', model='jinhybr/OCR-DocVQA-Donut')

    # Example questions
    questions = ['What is the policy number?', 'What is the coverage amount?', 'Who is the beneficiary?', 'What is the term period?']

    # Extract information from the insurance policy document image
    answers = {}
    for question in questions:
        result = doc_vqa(image_path=image_path, question=question)
        answers[question] = result['answer']

    return answers

# test_function_code --------------------

def test_extract_insurance_info():
    """
    Tests the extract_insurance_info function.
    """
    # Test with a sample image
    image_path = 'path/to/sample_image.jpg'
    result = extract_insurance_info(image_path)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert len(result) == 4, 'The result should contain answers to 4 questions.'
    for question, answer in result.items():
        assert isinstance(question, str), 'The question should be a string.'
        assert isinstance(answer, str), 'The answer should be a string.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_insurance_info()