# function_import --------------------

from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

# function_code --------------------

def summarize_text(text):
    """
    Summarizes a given long text using BigBird Pegasus model.

    Args:
        text (str): The long text to be summarized.

    Returns:
        str: The summarized text.
    """
    tokenizer = AutoTokenizer.from_pretrained('google/bigbird-pegasus-large-bigpatent')
    model = BigBirdPegasusForConditionalGeneration.from_pretrained('google/bigbird-pegasus-large-bigpatent')
    inputs = tokenizer(text, return_tensors='pt')
    prediction = model.generate(**inputs)
    summary = tokenizer.batch_decode(prediction)[0]
    return summary

# test_function_code --------------------

def test_summarize_text():
    """
    Tests the summarize_text function with some test cases.
    """
    test_text1 = 'This is a long text that needs to be summarized. It contains many details that are not necessary for understanding the main idea.'
    test_text2 = 'Another long text that needs summarization. It also contains many unnecessary details.'
    assert len(summarize_text(test_text1)) < len(test_text1)
    assert len(summarize_text(test_text2)) < len(test_text2)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_summarize_text()