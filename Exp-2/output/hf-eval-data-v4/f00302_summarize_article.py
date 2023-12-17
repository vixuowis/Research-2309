# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_article(article_text):
    """
    Summarize a long article using the PEGASUS model.

    Parameters:
    article_text (str): The text of the article to be summarized.

    Returns:
    str: The summarized version of the article.
    """
    # Load the PEGASUS summarization model
    summarizer = pipeline('summarization', model='google/pegasus-xsum')
    # Generate the summary
    summary = summarizer(article_text, min_length=75, max_length=150)[0]['summary_text']
    return summary

# test_function_code --------------------

def test_summarize_article():
    print("Testing summarize_article function.")
    # Example long article text
    article_text = "The quick brown fox jumps over the lazy dog. This is just a sample text that represents a longer article to demonstrate summarization."*10

    # Generate the summary
    summary = summarize_article(article_text)
    # Check that the summary is not empty and is shorter than the original article
    assert summary, "The summary should not be empty."
    assert len(summary) < len(article_text), "The summary should be shorter than the original article."
    print("Test passed!")

# Run the test function
test_summarize_article()