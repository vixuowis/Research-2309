# function_import --------------------

from transformers import BartTokenizer, BartModel

# function_code --------------------

def summarize_essay(essay: str) -> str:
    """
    Summarize a student's essay using the BART model from Hugging Face Transformers.

    Args:
        essay (str): The student's essay to be summarized.

    Returns:
        str: The summarized version of the essay.
    """
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartModel.from_pretrained('facebook/bart-base')
    inputs = tokenizer(essay, return_tensors='pt')
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states

# test_function_code --------------------

def test_summarize_essay():
    """
    Test the summarize_essay function.
    """
    essay = 'This is a test essay. It is only a test.'
    summary = summarize_essay(essay)
    assert isinstance(summary, str), 'The output should be a string.'
    assert len(summary) < len(essay), 'The summary should be shorter than the original essay.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_summarize_essay()