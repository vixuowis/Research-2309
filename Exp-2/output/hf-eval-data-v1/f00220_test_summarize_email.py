# Test function for summarize_email
# The function uses assert to check if the function works correctly
# The test function uses a sample email text and checks if the output is a string

def test_summarize_email():
    email_text = 'Long email text goes here...'
    summary = summarize_email(email_text)
    assert isinstance(summary, str), 'The function should return a string.'
    assert len(summary) < len(email_text), 'The summary should be shorter than the original text.'

test_summarize_email()