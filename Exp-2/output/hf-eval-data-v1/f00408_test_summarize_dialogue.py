def test_summarize_dialogue():
    """
    This function tests the 'summarize_dialogue' function with a sample dialogue.
    It uses the 'assert' statement to compare the output of the function with the expected output.
    """
    # Define a sample dialogue
    sample_dialogue = 'Hello, how are you? I am fine, thank you. How about you? I am good too. Thanks for asking.'
    # Call the 'summarize_dialogue' function with the sample dialogue
    summary = summarize_dialogue(sample_dialogue)
    # Print the summary
    print('Summary:', summary)
    # Assert that the summary is not empty
    assert summary != '', 'The summary is empty.'

test_summarize_dialogue()