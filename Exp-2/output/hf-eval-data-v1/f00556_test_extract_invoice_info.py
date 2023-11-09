# Test function for extract_invoice_info
# This function uses a sample invoice document and question to test the extract_invoice_info function.
def test_extract_invoice_info():
    # Sample invoice document and question
    doc_text = 'Sample invoice document text...'
    question = 'What is the total amount?'

    # Call the function with the sample invoice document and question
    answer = extract_invoice_info(doc_text, question)

    # Assert that the function returns a non-empty string
    assert isinstance(answer, str) and len(answer) > 0, 'The function should return a non-empty string.'

# Call the test function
test_extract_invoice_info()