def test_suggest_similar_questions():
    '''
    This function tests the suggest_similar_questions function.
    It uses a test dataset of questions and asserts that the function returns the expected results.
    '''
    # Define the test dataset
    user_question = 'What is your favorite color?'
    available_questions = ['What is your favorite food?', 'What is your favorite movie?', 'What is your favorite book?']
    
    # Call the function with the test dataset
    result = suggest_similar_questions(user_question, available_questions)
    
    # Assert that the function returns the expected result
    assert result in available_questions, f'Expected one of {available_questions}, but got {result}'
    
    # Print a success message
    print('All tests passed.')

# Run the test function
test_suggest_similar_questions()