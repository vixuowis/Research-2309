# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_text(text):
    """
    Summarize the input text using the PEGASUS model.

    Args:
        text (str): The text to summarize.

    Returns:
        str: The summarized text.

    Raises:
        ValueError: If the input text is not a string or is empty.
    """
    if not isinstance(text, str) or not text:
        raise ValueError('Input text must be a non-empty string.')
    summarizer = pipeline('summarization', model='google/pegasus-large')
    summary = summarizer(text)[0]['summary_text']
    return summary


# test_function_code --------------------

def test_summarize_text():
    print("Testing started.")
    # Assume that `sample_text` and `expected_summary` variables are defined

    # Test case 1: Non-empty text
    print("Testing case [1/3] started.")
    result_summary = summarize_text(sample_text)
    assert result_summary == expected_summary, f"Test case [1/3] failed: result_summary='{result_summary}', expected_summary='{expected_summary}'"

    # Test case 2: Empty text
    print("Testing case [2/3] started.")
    try:
        summarize_text('')
        assert False, "Test case [2/3] failed: ValueError not raised for empty text."
    except ValueError as e:
        assert str(e) == 'Input text must be a non-empty string.', f"Test case [2/3] failed: Exception message did not match."

    # Test case 3: Non-string input
    print("Testing case [3/3] started.")
    try:
        summarize_text(None)
        assert False, "Test case [3/3] failed: ValueError not raised for non-string input."
    except ValueError as e:
        assert str(e) == 'Input text must be a non-empty string.', f"Test case [3/3] failed: Exception message did not match."
    print("Testing finished.")


# call_test_function_line --------------------

test_summarize_text()