# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_invoice_information(image_path):
    """
    Extract information from an invoice image such as total amount, date of invoice, and name of the service provider.

    Args:
        image_path (str): The path to the invoice image file.

    Returns:
        dict: A dictionary with keys 'total_amount', 'date_of_invoice', and 'service_provider_name', containing the respective extracted information.
    """
    # Load the OCR DocVQA model using the pipeline function
    doc_vqa = pipeline('document-question-answering', model='jinhybr/OCR-DocVQA-Donut')

    # Define the questions to ask the model
    questions = {
        'total_amount': 'What is the total amount?',
        'date_of_invoice': 'What is the date of the invoice?',
        'service_provider_name': 'What is the name of the service provider?'
    }

    # Extract information for each question
    extracted_info = {}
    for key, question in questions.items():
        answer_data = doc_vqa(image_path=image_path, question=question)
        extracted_info[key] = answer_data['answer'] if 'answer' in answer_data else None

    return extracted_info

# test_function_code --------------------

def test_extract_invoice_information():
    print("Testing extract_invoice_information started.")
    # Assuming we have a sample invoice image for testing
    sample_image_path = 'sample_invoice.jpg'

    # Perform the information extraction
    extracted_info = extract_invoice_information(sample_image_path)

    # Test if the keys exist in the result dictionary
    print("Checking key existence.")
    assert all(key in extracted_info for key in ['total_amount', 'date_of_invoice', 'service_provider_name']), "Not all information keys are present."

    # Adding more specific tests if there are expected results from the sample
    # For example, if the known total amount of the invoice is '$1234.56'
    print("Testing total amount extraction.")
    assert extracted_info['total_amount'] == '$1234.56', "Incorrect total amount extracted."

    # Similarly, tests for date_of_invoice and service_provider_name can be added

    print("Testing extract_invoice_information finished.")

# Run the test
test_extract_invoice_information()