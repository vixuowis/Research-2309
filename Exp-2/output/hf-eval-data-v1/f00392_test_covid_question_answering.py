def test_covid_question_answering():
    """
    This function tests the covid_question_answering function.
    It uses a sample question and context, and checks if the returned answer is as expected.
    """
    question = 'What are the common symptoms of COVID-19?'
    context = 'COVID-19 is a respiratory disease with common symptoms such as cough, fever, and difficulty breathing.'
    expected_answer = 'cough, fever, and difficulty breathing'
    
    # Call the function with the test question and context
    answer = covid_question_answering(question, context)
    
    # Check if the returned answer is as expected
    assert answer.lower() in expected_answer.lower(), f'Expected {expected_answer}, but got {answer}'
    
    print('All tests passed.')

test_covid_question_answering()