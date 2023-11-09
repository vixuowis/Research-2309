def test_generate_story():
    """
    This function tests the generate_story function by comparing the output with the expected result.
    """
    # Test dataset
    prompts = ['Once upon a time in a small village...', 'In a galaxy far, far away...', 'In the heart of the city...']
    
    # Test the function with the test dataset
    for prompt in prompts:
        story = generate_story(prompt)
        
        # Using assert to check the function's output
        # Do not compare number strictly
        assert isinstance(story, str), 'The output should be a string.'
        assert len(story) > len(prompt), 'The generated story should be longer than the prompt.'

test_generate_story()