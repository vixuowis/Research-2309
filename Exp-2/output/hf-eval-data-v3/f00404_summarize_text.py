# function_import --------------------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# function_code --------------------

def summarize_text(article_text):
    """
    Summarize a given long article text.

    Args:
        article_text (str): The long article text to be summarized.

    Returns:
        str: The summarized text.
    """
    tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_summarizer')
    model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_summarizer')
    inputs = tokenizer.encode(article_text, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

# test_function_code --------------------

def test_summarize_text():
    """
    Test the summarize_text function.
    """
    article_text = 'This is a long article text that needs to be summarized.'
    summary_text = summarize_text(article_text)
    assert isinstance(summary_text, str), 'The result is not a string.'
    assert len(summary_text) < len(article_text), 'The summary is not shorter than the original text.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_summarize_text()