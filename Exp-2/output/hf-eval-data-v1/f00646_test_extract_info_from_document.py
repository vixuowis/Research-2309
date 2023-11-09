def test_extract_info_from_document():
    # Test the function with a sample OCR text and question
    ocr_text = "Invoice for: John Doe\nTotal amount due: $100"
    question = "What is the total amount due?"
    
    # Call the function with the sample OCR text and question
    answer = extract_info_from_document(ocr_text, question)
    
    # Assert that the function returns the correct answer
    assert answer == '$100', f'Error: {answer}'
    
    # Test the function with another sample OCR text and question
    ocr_text = "Invoice for: Jane Doe\nTotal amount due: $200"
    question = "What is the total amount due?"
    
    # Call the function with the sample OCR text and question
    answer = extract_info_from_document(ocr_text, question)
    
    # Assert that the function returns the correct answer
    assert answer == '$200', f'Error: {answer}'

test_extract_info_from_document()