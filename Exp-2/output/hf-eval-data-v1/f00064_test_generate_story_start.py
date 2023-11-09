def test_generate_story_start():
    """
    This function tests the 'generate_story_start' function.
    It uses a fixed prompt and checks if the output is a string.
    """
    prompt = 'A brave knight and a fearsome dragon'
    story_start = generate_story_start(prompt)
    
    # Check if the output is a string
    assert isinstance(story_start, str), 'The output should be a string.'
    
    # Check if the output is not empty
    assert len(story_start) > 0, 'The output should not be empty.'

test_generate_story_start()