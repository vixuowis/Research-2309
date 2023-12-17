# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_feedback(feedback):
    """
    Summarize customer feedback using a pre-trained summarization model.

    Args:
        feedback (str): The customer feedback text to be summarized.

    Returns:
        str: The summarized feedback.

    Raises:
        ValueError: If the `feedback` is empty or None.
    """
    if not feedback:
        raise ValueError('Feedback text is empty or None.')

    summarizer = pipeline('summarization', model='philschmid/bart-large-cnn-samsum')
    summary = summarizer(feedback, min_length=5, max_length=50)
    return summary[0]['summary_text']

# test_function_code --------------------

def test_summarize_feedback():
    print("Testing started.")
    sample_feedback = 'Customer service was great, but platform can be improved. Had issues with third-party tool integration.'

    print("Testing case [1/1] started.")
    summary = summarize_feedback(sample_feedback)
    assert isinstance(summary, str), f"Test case [1/1] failed: Expected a string summary, but got {type(summary).__name__}."
    assert 0 < len(summary) <= 50, f"Test case [1/1] failed: Summary length should be between 5 and 50, got {len(summary)}."
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_feedback()