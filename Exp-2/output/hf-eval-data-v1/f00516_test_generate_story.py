def test_generate_story():
    """
    This function tests the 'generate_story' function by generating a story with the starting phrase 'Once upon a time'.
    """
    starting_phrase = 'Once upon a time'
    
    # Generate the story
    generated_story = generate_story(starting_phrase)
    
    # Assert that the generated story starts with the starting phrase
    assert generated_story.startswith(starting_phrase), 'The generated story does not start with the starting phrase.'
    
    # Assert that the generated story is not just the starting phrase
    assert len(generated_story) > len(starting_phrase), 'The generated story is only the starting phrase.'

test_generate_story()