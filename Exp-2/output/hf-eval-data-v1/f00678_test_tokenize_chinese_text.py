def test_tokenize_chinese_text():
    """
    This function tests the 'tokenize_chinese_text' function.
    """
    # Define a test case
    test_text = '我爱自然语言处理'
    # Call the function with the test case
    tokens = tokenize_chinese_text(test_text)
    # Assert that the function returns a list
    assert isinstance(tokens, list), 'The function should return a list.'
    # Assert that the function does not return an empty list
    assert len(tokens) > 0, 'The function should not return an empty list.'

test_tokenize_chinese_text()