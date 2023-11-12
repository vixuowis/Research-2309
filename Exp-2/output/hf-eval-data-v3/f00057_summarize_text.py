# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_text(article: str, max_length: int = 130, min_length: int = 30, do_sample: bool = False) -> str:
    '''
    Summarizes a given text using the BART model fine-tuned on CNN Daily Mail.

    Args:
        article (str): The text to be summarized.
        max_length (int, optional): The maximum length of the summary. Defaults to 130.
        min_length (int, optional): The minimum length of the summary. Defaults to 30.
        do_sample (bool, optional): Whether to use sampling in generating the summary. Defaults to False.

    Returns:
        str: The summarized text.
    '''
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
    summary = summarizer(article, max_length=max_length, min_length=min_length, do_sample=do_sample)
    return summary[0]['summary_text']

# test_function_code --------------------

def test_summarize_text():
    '''
    Tests the summarize_text function.
    '''
    article = 'Apple Inc. reported its quarterly earnings results yesterday. The company posted a record-breaking revenue of $123.9 billion for the first quarter of 2022, up by 11% from the same period last year. The increase was fueled by stronger demand for iPhones, iPads, and Macs, as well as continued growth in its services segment.'
    summary = summarize_text(article)
    assert len(summary) <= 130 and len(summary) >= 30
    assert 'Apple Inc.' in summary
    assert 'quarterly earnings' in summary
    assert 'record-breaking revenue' in summary
    return 'All Tests Passed'

# call_test_function_code --------------------

test_summarize_text()