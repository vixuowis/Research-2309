def test_find_most_similar_sentences():
    '''
    This function tests the find_most_similar_sentences function.
    It uses a small set of sentences and checks if the function returns the expected result.
    '''
    sentences = ['I have a dog', 'My dog loves to play', 'There is a cat in our house', 'The cat and the dog get along well']
    expected_result = ('I have a dog', 'My dog loves to play')
    
    # Call the function with the test data
    result = find_most_similar_sentences(sentences)
    
    # Check if the result is as expected
    assert result == expected_result, f'Expected {expected_result}, but got {result}'
    
    print('All tests passed.')

test_find_most_similar_sentences()