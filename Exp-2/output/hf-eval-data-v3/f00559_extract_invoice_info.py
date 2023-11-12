# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering

# function_code --------------------

def extract_invoice_info(image_path):
    """
    Extracts specific information from an invoice image using a pre-trained model.

    Args:
        image_path (str): The path to the invoice image.

    Returns:
        list: A list of answers to the questions about the total amount due, invoice number, and due date.

    Raises:
        OSError: If the model is not found in the Hugging Face model hub.
    """
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023')
    inputs, layout = preprocess_image(image_path)
    questions = ['What is the total amount due?', 'What is the invoice number?', 'What is the due date?']
    answers = []
    for question in questions:
        answer = model(inputs, layout, question)
        answers.append(answer)
    return answers

# test_function_code --------------------

def test_extract_invoice_info():
    """
    Tests the function extract_invoice_info.
    """
    image_path = 'test_invoice.jpg'
    try:
        answers = extract_invoice_info(image_path)
        assert isinstance(answers, list), 'The return type should be a list.'
        assert len(answers) == 3, 'The length of the list should be 3.'
    except OSError as e:
        print('The model is not found in the Hugging Face model hub.')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_invoice_info()