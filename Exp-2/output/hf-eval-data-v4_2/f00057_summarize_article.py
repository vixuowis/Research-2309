# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_article(article, model='facebook/bart-large-cnn', max_length=130, min_length=30):
    """
    Summarize a given article using BART large model from Hugging Face Transformers.

    Args:
        article (str): The article text to summarize.
        model (str): The model to use for summarization. Default is 'facebook/bart-large-cnn'.
        max_length (int): The maximum length of the summary. Default is 130.
        min_length (int): The minimum length of the summary. Default is 30.

    Returns:
        str: The summarized text.

    Raises:
        ValueError: If the `article` is an empty string.
    """
    if not article:
        raise ValueError('The article provided is empty.')
    summarizer = pipeline('summarization', model=model)
    summary = summarizer(article, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# test_function_code --------------------

def test_summarize_article():
    print("Testing started.")
    # Testing with actual articles might require a transformer model to be set up for generating results,
    # which would take longer time to execute. Hence, using hypothetical outputs for demonstration.

    # Test case 1: Check for non-empty article.
    print("Testing case [1/1] started.")
    article = 'This is an example article text to be summarized. It should return a valid summary.'
    summary = summarize_article(article)
    assert summary, f"Test case [1/1] failed: The summary should not be empty for a non-empty article."
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_article()