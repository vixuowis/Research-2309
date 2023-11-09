def test_get_answer():
    '''
    This function tests the 'get_answer' function.
    It uses a sample question and context, and checks if the returned answer is correct.
    '''
    # Define a sample question and context
    question = 'What is the capital of France?'
    context = 'Paris is the capital of France.'
    
    # Get the answer from the 'get_answer' function
    answer = get_answer(question, context)
    
    # Check if the returned answer is correct
    assert answer == 'Paris', f'Error: {answer}'
    
    # Define another sample question and context
    question = 'Who is the president of the United States?'
    context = 'Joe Biden is the president of the United States.'
    
    # Get the answer from the 'get_answer' function
    answer = get_answer(question, context)
    
    # Check if the returned answer is correct
    assert answer == 'Joe Biden', f'Error: {answer}'

test_get_answer()