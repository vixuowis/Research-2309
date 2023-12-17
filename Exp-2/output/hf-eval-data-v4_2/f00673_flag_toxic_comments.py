# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def flag_toxic_comments(text):
    """
    Detect and flag toxic comments using a pre-trained model.

    Args:
        text (str): A string containing the comment to be evaluated.

    Returns:
        dict: A dictionary with keys 'label' and 'score' indicating the toxicity.

    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')

    toxic_comment_detector = pipeline(model='martin-ha/toxic-comment-model')
    return toxic_comment_detector(text)

# test_function_code --------------------

def test_flag_toxic_comments():
    print("Testing started.")
    # Test case 1: Normal comment
    print("Testing case [1/3] started.")
    result = flag_toxic_comments("Have a nice day!")
    assert 'label' in result and result['label'] == 'NON_TOXIC', f"Test case [1/3] failed: {result}"

    # Test case 2: Clearly toxic comment
    print("Testing case [2/3] started.")
    result = flag_toxic_comments("I hate you!")
    assert 'label' in result and result['label'] == 'TOXIC', f"Test case [2/3] failed: {result}"

    # Test case 3: Invalid input
    print("Testing case [3/3] started.")
    try:
        flag_toxic_comments(123)
        assert False, "Test case [3/3] failed: No exception raised for invalid input"
    except ValueError as e:
        assert str(e) == 'Input text must be a string.', f"Test case [3/3] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_flag_toxic_comments()