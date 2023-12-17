# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_scientific_article(article_text):
    """
    Summarize a scientific article using the google/pegasus-large model.

    Parameters:
        article_text (str): The text of the scientific article to be summarized.

    Returns:
        str: A concise summary of the article.
    """
    summarizer = pipeline('summarization', model='google/pegasus-large')
    summary = summarizer(article_text)
    return summary[0]['summary_text']

# test_function_code --------------------

def test_summarize_scientific_article():
    print("Testing started.")
    # Assume a hypothetical scientific article text
    article_text = "This is a sample scientific article text that explains the research done on the subject of natural language processing."

    # Test case 1: Check if the function returns a string
    print("Testing case [1/1] started.")
    summary = summarize_scientific_article(article_text)
    assert isinstance(summary, str), f"Test case [1/1] failed: Expected string, got {type(summary)}"
    print("Testing finished.")

# Run the test function
test_summarize_scientific_article()