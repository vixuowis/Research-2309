# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_scientific_article(article: str) -> str:
    """
    Summarize a scientific article using the 'google/pegasus-large' model from Hugging Face Transformers.

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
    Test the 'summarize_scientific_article' function.
    """
    article = "Here is the scientific article text..."
    summary = summarize_scientific_article(article)
    assert isinstance(summary, str), "The function should return a string."
    assert len(summary) > 0, "The function should return a non-empty string."

# call_test_function_code --------------------

test_summarize_scientific_article()