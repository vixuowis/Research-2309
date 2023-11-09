def test_get_correct_answer():
    '''
    This function tests the get_correct_answer function.
    It uses a sample summary text, question and options to test the function.
    The function asserts that the returned correct answer is in the options.
    '''
    summary_text = 'This is a summary of an article.'
    question = 'What is this a summary of?'
    options = ['A book', 'An article', 'A movie', 'A song']
    correct_answer = get_correct_answer(summary_text, question, options)
    assert correct_answer in options, f'Error: {correct_answer} not in options'

test_get_correct_answer()