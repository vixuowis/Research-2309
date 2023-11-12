# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_text(article: str, min_length: int = 75, max_length: int = 150) -> str:
    '''
    Summarize a given long text article using the PEGASUS model from Hugging Face Transformers.

    Args:
        article (str): The long article text to be summarized.
        min_length (int, optional): The minimum length of the summary. Defaults to 75.
        max_length (int, optional): The maximum length of the summary. Defaults to 150.

    Returns:
        str: The summarized text.
    '''
    summarizer = pipeline('summarization', model='google/pegasus-xsum')
    summary = summarizer(article, min_length=min_length, max_length=max_length)[0]['summary_text']
    return summary

# test_function_code --------------------

def test_summarize_text():
    '''
    Test the summarize_text function.
    '''
    article = 'This is a long article. It contains many details and information. It needs to be summarized.'
    summary = summarize_text(article)
    assert len(summary) <= 150 and len(summary) >= 75
    assert isinstance(summary, str)

    article = 'This is another long article. It also contains many details and information. It also needs to be summarized.'
    summary = summarize_text(article, min_length=50, max_length=100)
    assert len(summary) <= 100 and len(summary) >= 50
    assert isinstance(summary, str)

    return 'All Tests Passed'

# call_test_function_code --------------------

test_summarize_text()