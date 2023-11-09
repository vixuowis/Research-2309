def test_get_answer_from_textbook():
    '''
    This function tests the 'get_answer_from_textbook' function by using a sample question and textbook content.
    '''
    # Define a sample question and textbook content
    question = 'What is the function of mitochondria in a cell?'
    textbook_content = 'Mitochondria are the energy factories of the cell. They convert energy from food molecules into a useable form known as adenosine triphosphate (ATP).'
    
    # Get the answer to the question
    answer = get_answer_from_textbook(question, textbook_content)
    
    # Assert that the answer is correct
    assert answer == 'Mitochondria are the energy factories of the cell. They convert energy from food molecules into a useable form known as adenosine triphosphate (ATP).', 'Test failed: The answer is incorrect.'
    
    print('Test passed: The answer is correct.')

test_get_answer_from_textbook()