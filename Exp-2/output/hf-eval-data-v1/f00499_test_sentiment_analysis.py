def test_sentiment_analysis():
    """
    This function tests the 'sentiment_analysis' function with some example text.
    """
    # Example text
    text = 'The website text about technology'
    
    # Expected output (not exact, for illustrative purposes)
    expected_output = {'sequence': 'The website text about technology', 'labels': ['positive', 'negative'], 'scores': [0.6, 0.4]}
    
    # Get the actual output
    actual_output = sentiment_analysis(text)
    
    # Check that the actual output is as expected
    assert actual_output['sequence'] == expected_output['sequence']
    assert set(actual_output['labels']) == set(expected_output['labels'])
    assert abs(actual_output['scores'][0] - expected_output['scores'][0]) < 0.1
    assert abs(actual_output['scores'][1] - expected_output['scores'][1]) < 0.1

test_sentiment_analysis()