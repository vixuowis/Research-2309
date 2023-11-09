def test_summarize_news_article():
    '''
    This function tests the 'summarize_news_article' function.
    It uses a sample news article and checks if the output is a string.
    '''
    article_text = 'Videos that say approved vaccines are dangerous and cause autism, cancer or infertility are among those that will be taken down, the company said. The policy includes the termination of accounts of anti-vaccine influencers. Tech giants have been criticised for not doing more to counter false health information on their sites.'
    summary = summarize_news_article(article_text)
    assert isinstance(summary, str), 'The output should be a string.'

test_summarize_news_article()