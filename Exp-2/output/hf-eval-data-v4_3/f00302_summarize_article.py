# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

summarize_article(article_text):
    """
    Summarize a long article using the PEGASUS model from Hugging Face Transformers.

    Args:
        article_text (str): The text of the article to be summarized.

    Returns:
        str: A summary of the article.

    Raises:
        ValueError: If the article_text is empty.
    """
    if not article_text:
        raise ValueError('The article_text shall not be empty.')
    summarizer = pipeline('summarization', model='google/pegasus-xsum')
    summary = summarizer(article_text, min_length=75, max_length=150)[0]['summary_text']
    return summary


# test_function_code --------------------

def test_summarize_article():
    print("Testing started.")
    # Assuming sample_article is a predefined string with an appropriate length.
    sample_article = "Long article text here..."

    # Testing case [1/1] started
    print("Testing case [1/1] started.")
    summary = summarize_article(sample_article)
    assert summary, "Test case [1/1] failed: summary is empty."
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_article()