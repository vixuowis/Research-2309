def test_generate_natural_text():
    '''
    This function tests the generate_natural_text function.
    It uses a sample prompt and checks if the output is a string.
    '''
    # Sample prompt
    prompt = 'Hello, I am conscious and'
    # Call the function with the sample prompt
    generated_text = generate_natural_text(prompt)
    # Check if the output is a string
    assert isinstance(generated_text, str), 'The output should be a string.'
    # Check if the output is not empty
    assert len(generated_text) > 0, 'The output should not be empty.'

test_generate_natural_text()