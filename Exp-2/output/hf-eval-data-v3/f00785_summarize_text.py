# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_text(article, max_length=130, min_length=30, do_sample=False):
    '''
    Summarizes a given text using the Hugging Face Transformers library.

    Args:
        article (str): The text to be summarized.
        max_length (int, optional): The maximum length of the summary. Defaults to 130.
        min_length (int, optional): The minimum length of the summary. Defaults to 30.
        do_sample (bool, optional): Whether to use sampling in generating the summary. Defaults to False.

    Returns:
        str: The generated summary of the input text.
    '''
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
    summary = summarizer(article, max_length=max_length, min_length=min_length, do_sample=do_sample)[0]['summary_text']
    return summary

# test_function_code --------------------

def test_summarize_text():
    '''
    Tests the summarize_text function.
    '''
    article = 'In a shocking turn of events, the city council has decided to implement a new policy that will significantly impact the lives of the city residents. The policy, which was passed with a majority vote, will take effect from next month.'
    summary = summarize_text(article)
    assert isinstance(summary, str)
    assert len(summary) <= 130
    assert len(summary) >= 30
    return 'All Tests Passed'

# call_test_function_code --------------------

test_summarize_text()