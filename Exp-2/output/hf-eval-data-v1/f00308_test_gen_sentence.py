def test_gen_sentence():
    """
    This function tests the 'gen_sentence' function.
    It uses a set of test words and asserts that the output includes all of these words.
    """
    # Define the test words
    test_words = 'moon rabbit forest magic'
    # Generate a sentence using the test words
    test_sentence = gen_sentence(test_words)
    # Assert that each test word is in the generated sentence
    for word in test_words.split():
        assert word in test_sentence, f"Expected {word} to be in the generated sentence, but it was not."

# Run the test function
test_gen_sentence()