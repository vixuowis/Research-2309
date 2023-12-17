# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_news_article(article_text):
    """
    Summarize a news article using a pretrained BART model.

    Args:
        article_text (str): The text of the news article to be summarized.

    Returns:
        str: The summarized text.

    Raises:
        ValueError: If the 'article_text' is empty or None.
    """
    if not article_text:
        raise ValueError("The 'article_text' must not be empty.")

    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
    summary = summarizer(article_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    return summary

# test_function_code --------------------

def test_summarize_news_article():
    print("Testing started.")
    # Test case: Empty news article
    print("Testing case [1/2] started.")
    try:
        summarize_news_article('')
        assert False, "Test case [1/2] failed: Empty article did not raise ValueError."
    except ValueError as e:
        assert str(e) == "The 'article_text' must not be empty.", f"Test case [1/2] failed: {e}"

    # Test case: Actual news article
    print("Testing case [2/2] started.")
    test_article = "Breaking news: This is a test of the summarization pipeline."
    summary = summarize_news_article(test_article)
    assert summary, "Test case [2/2] failed: Failed to summarize the article."
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_news_article()