def test_generate_game_setting():
    """
    Test the generate_game_setting function.
    """
    initial_text = 'In a world filled with chaos and destruction'
    result = generate_game_setting(initial_text)
    assert isinstance(result, str), 'The result should be a string.'
    assert len(result) > len(initial_text), 'The generated text should be longer than the initial text.'

test_generate_game_setting()