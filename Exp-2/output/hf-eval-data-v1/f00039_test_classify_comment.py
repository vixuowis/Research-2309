# Test function for classify_comment
# This function uses a sample comment to test the classify_comment function
# The function asserts that the output of the classify_comment function is a list (as the pipeline function returns a list of dictionaries)
# The function does not compare numbers strictly as the output of the classify_comment function is a probability

def test_classify_comment():
    sample_comment = 'This is a test text.'
    result = classify_comment(sample_comment)
    assert isinstance(result, list), 'The output should be a list.'
    assert 'label' in result[0], 'Each item in the list should be a dictionary with a label key.'
    assert 'score' in result[0], 'Each item in the list should be a dictionary with a score key.'
    assert 0 <= result[0]['score'] <= 1, 'The score should be a probability between 0 and 1.'

test_classify_comment()