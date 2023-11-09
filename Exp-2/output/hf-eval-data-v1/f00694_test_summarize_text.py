def test_summarize_text():
    """
    This function tests the 'summarize_text' function by using a sample text.
    """
    # Define a sample text
    sample_text = 'The customer support service was excellent. All our concerns were attended to promptly by the friendly and knowledgeable staff. ...'
    
    # Call the 'summarize_text' function with the sample text
    summary = summarize_text(sample_text)
    
    # Assert that the summary is not empty
    assert len(summary) > 0, 'The summary is empty.'
    
    # Assert that the summary is indeed a summary of the sample text
    assert len(summary) < len(sample_text), 'The summary is not shorter than the original text.'

test_summarize_text()