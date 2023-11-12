# function_import --------------------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# function_code --------------------

def summarize_news_article(news_article):
    '''
    Summarize a news article using the Pegasus model.

    Args:
        news_article (str): The news article to summarize.

    Returns:
        str: The summarized news article.
    '''
    model_name = 'google/pegasus-cnn_dailymail'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer.encode(news_article, return_tensors='pt')
    summary_ids = model.generate(inputs)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_summarize_news_article():
    '''
    Test the summarize_news_article function.
    '''
    news_article = 'This is a test news article. It contains some information about a test event.'
    summary = summarize_news_article(news_article)
    assert isinstance(summary, str), 'The summary should be a string.'
    assert len(summary) < len(news_article), 'The summary should be shorter than the original article.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_summarize_news_article()