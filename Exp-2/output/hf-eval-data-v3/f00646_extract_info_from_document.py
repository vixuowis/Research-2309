# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_info_from_document(ocr_extracted_text: str, question: str) -> str:
    """
    Extracts relevant information from OCR extracted text using a question-answering model.

    Args:
        ocr_extracted_text (str): The text extracted from OCR.
        question (str): The question based on which relevant information needs to be extracted.

    Returns:
        str: The answer to the question based on the OCR extracted text.
    """
    qa_pipeline = pipeline('question-answering', model='tiennvcs/layoutlmv2-large-uncased-finetuned-vi-infovqa')
    answer = qa_pipeline({"context": ocr_extracted_text, "question": question})
    return answer['answer']

# test_function_code --------------------

def test_extract_info_from_document():
    """
    Tests the function extract_info_from_document.
    """
    ocr_extracted_text = 'This is a test document. The total amount due is $100.'
    question = 'What is the total amount due?'
    assert extract_info_from_document(ocr_extracted_text, question) == '$100'
    ocr_extracted_text = 'This is another test document. The due date is 30th June.'
    question = 'What is the due date?'
    assert extract_info_from_document(ocr_extracted_text, question) == '30th June'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_info_from_document()