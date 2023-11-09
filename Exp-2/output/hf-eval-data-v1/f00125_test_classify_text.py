def test_classify_text():
    """
    This function tests the 'classify_text' function with some sample data.
    """
    # Define the test data
    test_data = [
        'What is your name?',
        'My name is John.',
        'Where are you from?',
        'I am from New York.',
        'How old are you?',
        'I am 25 years old.'
    ]
    # Define the expected results
    expected_results = [
        'question',
        'statement',
        'question',
        'statement',
        'question',
        'statement'
    ]
    # Test the function with the test data
    for i, text in enumerate(test_data):
        assert classify_text(text) == expected_results[i], f'Error: {text} was classified incorrectly.'

# Run the test function
test_classify_text()