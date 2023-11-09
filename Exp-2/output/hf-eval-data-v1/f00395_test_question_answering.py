def test_question_answering():
    # Define a sample question and context
    question = 'What is the capital of France?'
    context = 'Paris is the capital of France.'

    # Call the function with the sample question and context
    answer = question_answering(question, context)

    # Assert that the function returns the correct answer
    assert answer == 'Paris', f'Error: {answer}'

    # Define a sample question with no answer in the context
    question = 'Who is the president of France?'
    context = 'Paris is the capital of France.'

    # Call the function with the sample question and context
    answer = question_answering(question, context)

    # Assert that the function returns an empty string for a question with no answer in the context
    assert answer == '', f'Error: {answer}'

test_question_answering()