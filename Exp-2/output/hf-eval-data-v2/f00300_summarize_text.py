# function_import --------------------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# function_code --------------------

def summarize_text(meeting_notes: str) -> str:
    """
    Summarizes the given text using the Pegasus model pretrained on the CNN/DailyMail dataset.

    Args:
        meeting_notes (str): The text to be summarized.

    Returns:
        str: The summarized text.
    """
    model_name = 'google/pegasus-cnn_dailymail'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer.encode(meeting_notes, return_tensors='pt')
    summary_ids = model.generate(inputs)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_summarize_text():
    """
    Tests the summarize_text function by summarizing a sample text and checking if the output is a string.
    """
    sample_text = 'This is a sample text for testing the summarization function.'
    summary = summarize_text(sample_text)
    assert isinstance(summary, str), 'The output should be a string.'

# call_test_function_code --------------------

test_summarize_text()