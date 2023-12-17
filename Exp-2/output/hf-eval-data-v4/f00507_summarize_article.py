# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_article(text):
    """
    Summarize the input article text using the PEGASUS model.

    Parameters:
        text (str): The article text to be summarized.

    Returns:
        str: The summarized version of the article.
    """
    summarizer = pipeline('summarization', model='google/pegasus-large')
    summary = summarizer(text, truncation=True)
    return summary[0]['summary_text']

# test_function_code --------------------

def test_summarize_article():
    print("Testing summarize_article function.")

    # Test case 1: Summarize a short article
    article_text = "The quick brown fox jumps over the lazy dog." * 10
    summary = summarize_article(article_text)
    expected_length = 130
    assert len(summary) <= expected_length, f"Test case 1 failed: summary length ({len(summary)}) exceeds expected length ({expected_length})."

    # Test case 2: Validate the output is a string
    assert isinstance(summary, str), "Test case 2 failed: The output is not a string."

    # Test case 3: Empty input
    summary = summarize_article("")
    expected_summary = ""
    assert summary == expected_summary, "Test case 3 failed: Empty text did not return an empty summary."

    print("All tests passed.")

test_summarize_article()