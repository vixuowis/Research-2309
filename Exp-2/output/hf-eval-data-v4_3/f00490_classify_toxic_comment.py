# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_toxic_comment(comment):
    """
    Classify a user comment as toxic or non-toxic.

    Args:
        comment (str): A user-generated comment to be assessed.

    Returns:
        A dictionary with the toxicity score and label.

    Raises:
        ValueError: If the comment is not a string or is empty.
    """
    if not isinstance(comment, str) or not comment:
        raise ValueError('Comment must be a non-empty string.')
    toxic_classifier = pipeline(model='martin-ha/toxic-comment-model')
    results = toxic_classifier(comment)
    return ({'score': results[0]['score'], 'label': results[0]['label']})

# test_function_code --------------------

def test_classify_toxic_comment():
    print("Testing started.")
    # Test case 1: Testing a clearly non-toxic comment
    print("Testing case [1/3] started.")
    assert classify_toxic_comment("Have a great day!")['label'] == 'NON_TOXIC', "Test case [1/3] failed: 'Have a great day!' should be classified as NON_TOXIC"

    # Test case 2: Testing a clearly toxic comment
    print("Testing case [2/3] started.")
    assert classify_toxic_comment("You are a horrible person!")['label'] == 'TOXIC', "Test case [2/3] failed: 'You are a horrible person!' should be classified as TOXIC"

    # Test case 3: Testing input type validation
    print("Testing case [3/3] started.")
    try:
        classify_toxic_comment(123)
        assert False, "Test case [3/3] failed: Integer input should raise ValueError"
    except ValueError:
        pass
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_toxic_comment()