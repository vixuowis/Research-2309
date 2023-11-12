# function_import --------------------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# function_code --------------------

def summarize_text(meeting_notes: str) -> str:
    '''
    Summarizes the given text using the Pegasus model from Hugging Face Transformers.

    Args:
        meeting_notes (str): The text to be summarized.

    Returns:
        str: The summarized text.
    '''
    model_name = 'google/pegasus-cnn_dailymail'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer.encode(meeting_notes, return_tensors='pt')
    summary_ids = model.generate(inputs)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_summarize_text():
    '''
    Tests the summarize_text function.
    '''
    text1 = 'This is a long text that needs to be summarized. It contains many details that are not necessary for understanding the main idea.'
    text2 = 'Another long text that needs to be summarized. It also contains many unnecessary details.'
    assert len(summarize_text(text1)) < len(text1)
    assert len(summarize_text(text2)) < len(text2)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_summarize_text()