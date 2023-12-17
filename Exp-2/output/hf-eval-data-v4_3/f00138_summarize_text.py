# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import BartTokenizer, BartForConditionalGeneration

# function_code --------------------

def summarize_text(input_text):
    """
    Summarizes the given text using the DistilBART model from Hugging Face.

    Args:
        input_text (str): The text to be summarized.

    Returns:
        str: The summarized text.

    Raises:
        ValueError: If the input text is empty.
    """
    if not input_text:
        raise ValueError('Input text is empty')

    tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
    model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
    inputs = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_summarize_text():
    print("Testing started.")
    # Test case: test summarization on a simple text
    print("Testing case [1/1] started.")
    input_text = "This is a long article that needs to be summarized to grasp the main points."
    expected_summary = "This is summarized text."
    actual_summary = summarize_text(input_text)
    assert expected_summary in actual_summary, f"Test case [1/1] failed: expected summary to contain '"{expected_summary}"', got '"{actual_summary}"'."
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_text()