# requirements_file --------------------

!pip install -U transformers, torch, datasets, tokenizers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_information_from_document(ocr_extracted_text, question):
    # Initialize the pipeline for question-answering with the specified model
    qa_pipeline = pipeline('question-answering', model='tiennvcs/layoutlmv2-large-uncased-finetuned-vi-infovqa')

    # Use the pipeline to retrieve the answer from the document based on the OCR text and the question
    answer = qa_pipeline({'context': ocr_extracted_text, 'question': question})

    # Return the answer obtained
    return answer


# test_function_code --------------------

def test_extract_information_from_document():
    print("Testing started.")
    ocr_extracted_text = 'This is a sample document text where the total amount due is $1234.56.'
    question = 'What is the total amount due?'

    # Test case 1: Check if the function correctly extracts the total amount due
    print("Testing case [1/1] started.")
    result = extract_information_from_document(ocr_extracted_text, question)
    assert result['answer'] == '$1234.56', f"Test case [1/1] failed: Expected '$1234.56', got {result['answer']}"
    print("Testing finished.")
