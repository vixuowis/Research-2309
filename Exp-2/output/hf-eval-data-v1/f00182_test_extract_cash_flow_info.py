def test_extract_cash_flow_info():
    # Define a sample question and financial document
    question = 'What was the net cash flow from operating activities?'
    financial_document = '...'
    
    # Call the function with the sample inputs
    answer = extract_cash_flow_info(question, financial_document)
    
    # Assert that the function returns a string (the answer should always be a string)
    assert isinstance(answer, str), 'The function should return a string.'
    
    # Assert that the function does not return an empty string (the answer should not be empty)
    assert answer != '', 'The function should not return an empty string.'

test_extract_cash_flow_info()