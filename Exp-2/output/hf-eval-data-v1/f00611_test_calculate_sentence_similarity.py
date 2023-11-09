# Test function for calculate_sentence_similarity
# This function tests the calculate_sentence_similarity function using a set of test sentences.
# It uses the assert statement to check if the calculated similarity is within a certain range, as the similarity is not expected to be exactly equal due to the nature of the calculation.
def test_calculate_sentence_similarity():
    # Define the test sentences
    sentence1 = 'What time is it?'
    sentence2 = 'Can you tell me the current time?'
    # Call the calculate_sentence_similarity function
    similarity = calculate_sentence_similarity(sentence1, sentence2)
    # Check if the calculated similarity is within the expected range
    assert 0.8 <= similarity <= 1.0, 'Test failed: The calculated similarity is not within the expected range.'
    print('Test passed.')

# Call the test function
test_calculate_sentence_similarity()