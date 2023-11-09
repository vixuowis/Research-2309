def test_summarize_text():
    # Test the summarize_text function with a sample text
    article_text = 'Videos that say approved vaccines are dangerous and cause autism, cancer or infertility are among those that will be taken down, the company said. The policy includes the termination of accounts of anti-vaccine influencers. Tech giants have been criticised for not doing more to counter false health information on their sites.'
    summary = summarize_text(article_text)
    assert isinstance(summary, str), 'The output should be a string.'
    assert len(summary) > 0, 'The output should not be empty.'

test_summarize_text()