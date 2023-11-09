def test_summarize_text():
    """
    This function tests the 'summarize_text' function.
    It uses a sample text and checks if the output is a string (as expected).
    """
    # Sample text
    sample_text = "This is a long text that needs to be summarized. It contains many details that may not be necessary for a brief understanding of the topic."

    # Call the 'summarize_text' function
    summary = summarize_text(sample_text)

    # Check if the output is a string
    assert isinstance(summary, str), 'The output should be a string.'

    # Check if the output is shorter than the input text
    assert len(summary) < len(sample_text), 'The output should be shorter than the input text.'

test_summarize_text()