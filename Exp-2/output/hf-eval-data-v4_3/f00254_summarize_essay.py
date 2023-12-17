# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import BartTokenizer, BartModel

# function_code --------------------

def summarize_essay(essay_text):
    """
    Summarizes the input essay text using BART model.

    Args:
        essay_text (str): The essay text to be summarized.

    Returns:
        str: The summary of the essay.

    Raises:
        ValueError: If the `essay_text` is not a valid string.
    """
    if not isinstance(essay_text, str):
        raise ValueError('Input essay_text must be a valid string.')
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartModel.from_pretrained('facebook/bart-base')
    inputs = tokenizer(essay_text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# test_function_code --------------------

def test_summarize_essay():
    print('Testing started.')

    # Test case 1: Regular essay text
    print('Testing case [1/3] started.')
    essay_text = 'Recently, there has been an increase in the interest of natural language processing.'
    summary = summarize_essay(essay_text)
    assert summary, 'Test case [1/3] failed: The summary should not be empty.'

    # Test case 2: Empty string
    print('Testing case [2/3] started.')
    essay_text = ''
    try:
        summarize_essay(essay_text)
        assert False, 'Test case [2/3] failed: ValueError expected.'
    except ValueError as e:
        assert str(e) == 'Input essay_text must be a valid string.', 'Test case [2/3] failed: Unexpected ValueError message.'

    # Test case 3: Non-string input
    print('Testing case [3/3] started.')
    essay_text = 123
    try:
        summarize_essay(essay_text)
        assert False, 'Test case [3/3] failed: ValueError expected.'
    except ValueError as e:
        assert str(e) == 'Input essay_text must be a valid string.', 'Test case [3/3] failed: Unexpected ValueError message.'

    print('Testing finished.')

# call_test_function_line --------------------

test_summarize_essay()