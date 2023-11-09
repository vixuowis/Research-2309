def test_generate_poem():
    """
    This function tests the 'generate_poem' function by providing a prompt and checking the output.
    """
    # Define the prompt
    prompt = 'Once upon a time, in a land of greenery and beauty,'
    # Generate the poem
    poem = generate_poem(prompt)
    # Check that the output is a string
    assert isinstance(poem, str), 'The output should be a string.'
    # Check that the output is not empty
    assert len(poem) > 0, 'The output should not be empty.'
    # Check that the output contains the prompt
    assert prompt in poem, 'The output should contain the prompt.'

test_generate_poem()