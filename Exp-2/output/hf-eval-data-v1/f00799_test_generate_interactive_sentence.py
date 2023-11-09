def test_generate_interactive_sentence():
    """
    Test the function generate_interactive_sentence.
    """
    test_sentence = 'Tell me more about your [MASK] hobbies.'
    expected_output = 'Tell me more about your favorite hobbies.'
    assert generate_interactive_sentence(test_sentence) == expected_output

test_generate_interactive_sentence()