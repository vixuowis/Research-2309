# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering

# function_code --------------------

def extract_invoice_info(image_path):
    """
    Extracts specific information from an invoice image using a pre-trained model.

    Args:
        image_path (str): The path to the invoice image.

    Returns:
        dict: A dictionary containing the total amount due, invoice number, and due date.
    """
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023')
    inputs, layout = preprocess_image(image_path)
    questions = ['What is the total amount due?', 'What is the invoice number?', 'What is the due date?']
    answers = {}
    for question in questions:
        answer = model(inputs, layout, question)
        answers[question] = answer
    return answers

# test_function_code --------------------

def test_extract_invoice_info():
    """
    Tests the extract_invoice_info function.
    """
    image_path = 'test_invoice.jpg'  # replace with path to your test invoice image
    answers = extract_invoice_info(image_path)
    assert isinstance(answers, dict), 'The return type should be a dictionary.'
    assert len(answers) == 3, 'The dictionary should contain three items.'
    assert 'What is the total amount due?' in answers, 'The total amount due should be in the answers.'
    assert 'What is the invoice number?' in answers, 'The invoice number should be in the answers.'
    assert 'What is the due date?' in answers, 'The due date should be in the answers.'

# call_test_function_code --------------------

test_extract_invoice_info()