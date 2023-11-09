def test_classify_topic():
    '''
    This function tests the classify_topic function.
    It uses assert to verify the function's output with the expected output.
    '''
    # Define the test sentence and the expected output
    test_sentence = 'Apple just announced the newest iPhone X'
    expected_output = 'technology'
    # Call the classify_topic function
    output = classify_topic(test_sentence)
    # Assert that the output is as expected
    assert output == expected_output, f'Expected {expected_output}, but got {output}'
    
    # Define another test sentence and the expected output
    test_sentence = 'The universe is expanding at an accelerating rate'
    expected_output = 'science'
    # Call the classify_topic function
    output = classify_topic(test_sentence)
    # Assert that the output is as expected
    assert output == expected_output, f'Expected {expected_output}, but got {output}'
    
    # Define another test sentence and the expected output
    test_sentence = 'Shakespeare was a great playwright'
    expected_output = 'literature'
    # Call the classify_topic function
    output = classify_topic(test_sentence)
    # Assert that the output is as expected
    assert output == expected_output, f'Expected {expected_output}, but got {output}'

test_classify_topic()