# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_text(article_text):
    """
    Summarize the given text using PEGASUS model.

    Args:
        article_text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    """
    summarizer = pipeline('summarization', model='google/pegasus-large')
    summary = summarizer(article_text)
    return summary[0]['summary_text']

# test_function_code --------------------

def test_summarize_text():
    """
    Test the summarize_text function.
    """
    text1 = 'This is a long text that needs to be summarized. It contains many details that are not necessary for understanding the main idea.'
    text2 = 'Another long text that needs to be summarized. It also contains many unnecessary details.'
    text3 = 'A third long text that needs to be summarized. It is full of details that are not important for the main idea.'

    assert len(summarize_text(text1)) < len(text1)
    assert len(summarize_text(text2)) < len(text2)
    assert len(summarize_text(text3)) < len(text3)

    print('All Tests Passed')

# call_test_function_code --------------------

test_summarize_text()