def test_question_answering():
    # Define the question and context
    question = 'Why is model conversion important?'
    context = 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    # Call the function with the test data
    result = question_answering(question, context)
    # Assert that the function returns a dictionary (the expected output format)
    assert isinstance(result, dict), 'The function should return a dictionary.'
    # Assert that the dictionary contains the 'answer' key
    assert 'answer' in result, 'The result dictionary should contain the \'answer\' key.'
    # Assert that the answer is a string
    assert isinstance(result['answer'], str), 'The answer should be a string.'

test_question_answering()