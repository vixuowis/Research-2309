def test_summarize_russian_text():
    # Define a Russian text for testing
    russian_text = 'Пример оригинального русского текста здесь...'
    
    # Generate a summary of the text
    summary = summarize_russian_text(russian_text)
    
    # Print the summary
    print(summary)
    
    # Assert that the summary is not None
    assert summary is not None
    
    # Assert that the summary is a string
    assert isinstance(summary, str)

test_summarize_russian_text()