def test_get_answer():
    """
    This function tests the 'get_answer' function.
    """
    # Define a question and text
    question = 'What has Huggingface done ?'
    text = 'Huggingface has democratized NLP. Huge thanks to Huggingface for this.'
    
    # Get the answer to the question
    answer = get_answer(question, text)
    
    # Assert that the answer is correct
    assert answer == 'democratized NLP', 'Test failed: The answer is incorrect.'
    
    print('Test passed.')

test_get_answer()