# requirements_file --------------------

!pip install -U transformers>=4.11.0

# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering

# function_code --------------------

def extract_invoice_information(image_path):
    """
    Extract specific information from an invoice image including total amount due,
    invoice number, and due date using a pre-trained document question answering model.

    Parameters:
        image_path (str): The path to the invoice image.

    Returns:
        dict: A dictionary containing extracted information with keys 'total_amount_due',
        'invoice_number', and 'due_date'.
    """
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023')
    inputs, layout = preprocess_image(image_path)  # A custom function to preprocess the image
    questions = ['What is the total amount due?', 'What is the invoice number?', 'What is the due date?']
    answers = {}
    for question in questions:
        answer = model(inputs, layout, question)
        if question == 'What is the total amount due?':
            answers['total_amount_due'] = answer
        elif question == 'What is the invoice number?':
            answers['invoice_number'] = answer
        elif question == 'What is the due date?':
            answers['due_date'] = answer
    return answers

# test_function_code --------------------

def test_extract_invoice_information():
    print("Testing extract_invoice_information function.")
    image_path = 'sample_invoice.jpg'  # Replace with your sample invoice image

    # Call the function with the sample invoice
    extracted_info = extract_invoice_information(image_path)

    # Test if the function returns the right keys
    assert 'total_amount_due' in extracted_info, "Key 'total_amount_due' missing in the result"
    assert 'invoice_number' in extracted_info, "Key 'invoice_number' missing in the result"
    assert 'due_date' in extracted_info, "Key 'due_date' missing in the result"

    print("All tests passed!")

# Run the tests
test_extract_invoice_information()