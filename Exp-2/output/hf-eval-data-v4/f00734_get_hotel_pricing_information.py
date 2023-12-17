# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_hotel_pricing_information(question, pricing_document):
    """
    This function takes a question related to hotel pricing and the
    hotel's pricing document context, and returns the answer using
    the LayoutLM model for document question answering.

    :param question: The question string regarding hotel pricing.
    :param pricing_document: The text content of the hotel pricing document.
    :return: The answer to the question from the hotel pricing document.
    """
    # Initialize the document question-answering model
    document_qa_model = pipeline('question-answering', model='pardeepSF/layoutlm-vqa')

    # Get the answer from the pricing document
    answer = document_qa_model(question=question, context=pricing_document)
    return answer

# test_function_code --------------------

def test_get_hotel_pricing_information():
    print("Testing get_hotel_pricing_information() started.")

    # A dummy pricing document for test purpose
    pricing_document = "Deluxe Suite: $250 per night. Standard Room: $150 per night."

    # Test case 1: Ask about the Deluxe Suite
    question_deluxe = "What is the cost of a deluxe suite per night?"
    answer_deluxe = get_hotel_pricing_information(question_deluxe, pricing_document)
    assert answer_deluxe['answer'] == '$250', f"Test case [1/2] failed: {answer_deluxe}"
    print("Test case [1/2] succeeded.")

    # Test case 2: Ask about the Standard Room
    question_standard = "What is the cost of a standard room per night?"
    answer_standard = get_hotel_pricing_information(question_standard, pricing_document)
    assert answer_standard['answer'] == '$150', f"Test case [2/2] failed: {answer_standard}"
    print("Test case [2/2] succeeded.")

    print("Testing finished.")

# Run the test function
test_get_hotel_pricing_information()