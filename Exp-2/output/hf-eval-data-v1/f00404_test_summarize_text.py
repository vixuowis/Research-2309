def test_summarize_text():
    '''
    This function tests the summarize_text function.
    It uses a sample text and checks if the output is a string.
    '''
    # Sample text
    article_text = 'India wicket-keeper batsman Rishabh Pant has said someone from the crowd threw a ball on pacer Mohammed Siraj while he was fielding in the ongoing third Test against England on Wednesday. Pant revealed the incident made India skipper Virat Kohli upset.'

    # Call the summarize_text function
    summary_text = summarize_text(article_text)

    # Check if the output is a string
    assert isinstance(summary_text, str), 'The output should be a string.'

    # Check if the output is not empty
    assert len(summary_text) > 0, 'The output should not be empty.'

test_summarize_text()