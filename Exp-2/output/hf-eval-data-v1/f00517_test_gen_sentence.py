def test_gen_sentence():
    """
    This function tests the 'gen_sentence' function by generating a sentence using a list of words and asserting that the output is a string.
    """
    # Define a list of words to be included in the sentence
    words = 'tree plant ground hole dig'
    
    # Generate a sentence using the 'gen_sentence' function
    generated_sentence = gen_sentence(words)
    
    # Assert that the output is a string
    assert isinstance(generated_sentence, str), 'The output should be a string.'
    
    # Assert that the output is not empty
    assert len(generated_sentence) > 0, 'The output should not be empty.'
    
    # Assert that the output contains the input words
    for word in words.split():
        assert word in generated_sentence, f'The output should contain the word {word}.'

test_gen_sentence()