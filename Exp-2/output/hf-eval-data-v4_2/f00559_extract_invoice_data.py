# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering

# function_code --------------------

def extract_invoice_data(image_path: str):
    """
    Extract specific information from an invoice image, such as total amount due,
    invoice number, and due date.

    Args:
        image_path (str): The file path to the invoice image.

    Returns:
        dict: Extracted information containing total amount due, invoice number,
        and due date.

    Raises:
        FileNotFoundError: If the invoice image file does not exist.
    """
    # Verify if the image file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The file {image_path} does not exist.")

    # Load the pre-trained model
    model = AutoModelForDocumentQuestionAnswering.from_pretrained(
        'L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023')

    # Preprocess the image (This function should be implemented based on your dataset)
    inputs, layout = preprocess_image(image_path)  # A custom function to preprocess the image

    # Define the questions to ask the model
    questions = ['What is the total amount due?', 'What is the invoice number?', 'What is the due date?']
    answers = {}

    # Query the model for each question
    for question in questions:
        answer = model(inputs, layout, question)
        answers[question] = answer

    # Return the extracted data
    return answers

# test_function_code --------------------

def test_extract_invoice_data():
    print("Testing started.")

    # Assuming preprocess_image and a sample dataset are available
    sample_image = 'sample_invoice.jpg'  # Sample invoice image for testing

    # Testing case 1: Check if FileNotFoundError is raised for non-existent file
    print("Testing case [1/3] started.")
    try:
        extract_invoice_data('non_existent_file.jpg')
        assert False, "Test case [1/3] failed: FileNotFoundError not raised for non-existent file."
    except FileNotFoundError:
        pass  # Expected exception

    # Testing case 2: Check for correct keys in the output
    print("Testing case [2/3] started.")
    extracted_data = extract_invoice_data(sample_image)
    assert set(extracted_data.keys()) == set(['What is the total amount due?', 'What is the invoice number?', 'What is the due date?']), "Test case [2/3] failed: Output keys do not match expected keys."

    # Testing case 3: Check for valid output types
    print("Testing case [3/3] started.")
    assert all(isinstance(value, str) for value in extracted_data.values()), "Test case [3/3] failed: Output values are not all strings."
    print("Testing finished.")


# call_test_function_line --------------------

test_extract_invoice_data()