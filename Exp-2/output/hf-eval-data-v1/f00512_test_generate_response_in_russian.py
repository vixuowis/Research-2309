def test_generate_response_in_russian():
    '''
    This function tests the generate_response_in_russian function.
    It uses a sample input text and checks if the function returns a list of responses.
    '''
    # Define a sample input text
    input_text = '@@ПЕРВЫЙ@@ привет @@ВТОРОЙ@@ привет @@ПЕРВЫЙ@@ как дела? @@ВТОРОЙ@@'
    # Call the function with the sample input text
    responses = generate_response_in_russian(input_text)
    # Check if the function returns a list
    assert isinstance(responses, list), 'The function should return a list.'
    # Check if the list contains responses
    assert len(responses) > 0, 'The list should contain at least one response.'

test_generate_response_in_russian()