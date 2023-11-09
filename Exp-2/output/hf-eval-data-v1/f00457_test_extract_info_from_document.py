def test_extract_info_from_document():
    """
    This function tests the extract_info_from_document function.
    It uses a sample document and a set of questions, and checks if the function returns answers.
    """
    # Define a sample document path and a set of questions
    document_path = 'path/to/sample/document'
    questions = ['What is the total amount?', 'When is the due date?']
    
    # Call the function with the sample inputs
    answers = extract_info_from_document(document_path, questions)
    
    # Check if the function returns answers for all questions
    for question in questions:
        assert question in answers, f'No answer returned for question: {question}'
        assert isinstance(answers[question], str), f'Answer for question: {question} is not a string'
    
    print('All tests passed.')

# Run the test function
test_extract_info_from_document()