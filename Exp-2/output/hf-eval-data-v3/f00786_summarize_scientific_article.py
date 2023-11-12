# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_scientific_article(article):
    """
    Summarize a scientific article using the 'google/pegasus-large' model.

    Args:
        article (str): The text of the scientific article to be summarized.

    Returns:
        str: The summary of the article.
    """
    summarizer = pipeline('summarization', model='google/pegasus-large')
    summary = summarizer(article)
    return summary

# test_function_code --------------------

def test_summarize_scientific_article():
    """
    Test the function 'summarize_scientific_article'.
    """
    article = "This is a test article. It contains some important information."
    summary = summarize_scientific_article(article)
    assert isinstance(summary, str), 'The result should be a string.'
    assert len(summary) < len(article), 'The summary should be shorter than the original article.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_summarize_scientific_article()