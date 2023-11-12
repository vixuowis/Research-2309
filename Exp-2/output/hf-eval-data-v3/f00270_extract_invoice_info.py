# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_invoice_info(image_path: str, questions: list) -> list:
    '''
    Extracts information from an invoice image using a document-question-answering model.

    Args:
        image_path (str): The path to the invoice image.
        questions (list): A list of questions to ask the model.

    Returns:
        list: A list of answers from the model.
    '''
    doc_vqa = pipeline('document-question-answering', model='jinhybr/OCR-DocVQA-Donut')
    answers = [doc_vqa(image_path=image_path, question=q) for q in questions]
    return answers

# test_function_code --------------------

def test_extract_invoice_info():
    '''
    Tests the function extract_invoice_info.
    '''
    image_path = 'path/to/invoice_image.jpg'
    questions = ['What is the total amount?', 'What is the date of the invoice?', 'What is the name of the service provider?']
    answers = extract_invoice_info(image_path, questions)
    assert isinstance(answers, list), 'The result is not a list.'
    assert len(answers) == len(questions), 'The number of answers does not match the number of questions.'
    for answer in answers:
        assert isinstance(answer, str), 'The answer is not a string.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_invoice_info()