# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_scientific_article(article_text):
    """
    Summarize a scientific article using the Pegasus-large model.

    Args:
        article_text (str): The text of the scientific article to be summarized.

    Returns:
        str: The summarized text.

    Raises:
        ValueError: If the article_text is not a string or is empty.
    """
    if not article_text or not isinstance(article_text, str):
        raise ValueError('The article_text must be a non-empty string.')
    summarizer = pipeline('summarization', model='google/pegasus-large')
    summary = summarizer(article_text, max_length=1024, min_length=5, length_penalty=2.0, no_repeat_ngram_size=3)
    return summary[0]['summary_text']

# test_function_code --------------------

def test_summarize_scientific_article():
    print("Testing started.")
    sample_article = "Recent advancements in AI have led to improvements in machine learning models..."

    # Testing case 1: Non-empty string
    print("Testing case [1/2] started.")
    summary = summarize_scientific_article(sample_article)
    assert len(summary) > 0, "Test case [1/2] failed: The summary should not be empty."

    # Testing case 2: Handling empty string
    print("Testing case [2/2] started.")
    try:
        summarize_scientific_article('')
        assert False, "Test case [2/2] failed: ValueError expected"
    except ValueError as e:
        assert str(e) == 'The article_text must be a non-empty string.', "Test case [2/2] failed: Incorrect error message."
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_scientific_article()