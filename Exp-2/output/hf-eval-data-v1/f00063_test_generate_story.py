def test_generate_story():
    """
    This function tests the generate_story function.
    """
    # Define a test prompt
    test_prompt = 'Write a story about a spaceship journey to a distant planet in search of a new home for humanity.'
    
    # Generate a story using the test prompt
    test_story = generate_story(test_prompt)
    
    # Assert that the story is not empty
    assert len(test_story) > 0, 'The generated story is empty.'
    
    # Assert that the story starts with the test prompt
    assert test_story.startswith(test_prompt), 'The generated story does not start with the test prompt.'
    
    # Assert that the story is not exactly the same as the test prompt
    assert test_story != test_prompt, 'The generated story is exactly the same as the test prompt.'

test_generate_story()