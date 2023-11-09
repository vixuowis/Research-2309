def test_get_answer_from_book():
    """
    This function tests the 'get_answer_from_book' function.
    It uses a sample context and question, and checks if the returned answer is a string.
    """
    # Define a sample context and question
    context = 'This is a book about the history of the world.'
    question = 'What is this book about?'
    
    # Call the function with the sample context and question
    answer = get_answer_from_book(context, question)
    
    # Check if the returned answer is a string
    assert isinstance(answer, str), 'The function should return a string.'
    
    # Check if the returned answer is not empty
    assert answer != '', 'The function should return a non-empty string.'

test_get_answer_from_book()