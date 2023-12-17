# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# function_code --------------------

def classify_text(text):
    """
    Classifies a given text as a question or a statement.

    Args:
        text (str): The text to be classified.

    Returns:
        str: A string indicating whether the text is a 'question' or 'statement'.

    Raises:
        ValueError: If the text input is empty or not a string.
    """
    if not isinstance(text, str) or not text:
        raise ValueError('Input text must be a non-empty string.')

    tokenizer = AutoTokenizer.from_pretrained('shahrukhx01/question-vs-statement-classifier')
    model = AutoModelForSequenceClassification.from_pretrained('shahrukhx01/question-vs-statement-classifier')

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(dim=-1).item()

    return 'question' if predicted_class == 0 else 'statement'

# test_function_code --------------------

def test_classify_text():
    print("Testing started.")

    # Test case 1: Input is a question
    print("Testing case [1/3] started.")
    assert classify_text("Is this a question?") == 'question', "Test case [1/3] failed: Expected 'question'"

    # Test case 2: Input is a statement
    print("Testing case [2/3] started.")
    assert classify_text("This is a statement.") == 'statement', "Test case [2/3] failed: Expected 'statement'"

    # Test case 3: Input is invalid
    print("Testing case [3/3] started.")
    try:
        classify_text("")  # Empty string
        assert False, "Test case [3/3] failed: ValueError expected"
    except ValueError as e:
        assert str(e) == 'Input text must be a non-empty string.', "Test case [3/3] failed: Unexpected ValueError message"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_text()