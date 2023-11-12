# function_import --------------------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# function_code --------------------

def summarize_text(email_text):
    '''
    Summarizes the given text using the PEGASUS model.

    Args:
        email_text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    '''
    model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_summarizer')
    tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_summarizer')
    input_ids = tokenizer(email_text, return_tensors='pt').input_ids
    summary_ids = model.generate(input_ids)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

# test_function_code --------------------

def test_summarize_text():
    '''
    Tests the summarize_text function.
    '''
    test_text = 'This is a long email text that needs to be summarized.'
    summary = summarize_text(test_text)
    assert isinstance(summary, str), 'The result is not a string.'
    assert len(summary) < len(test_text), 'The summary is not shorter than the original text.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_summarize_text()