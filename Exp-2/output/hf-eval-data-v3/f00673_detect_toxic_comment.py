# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_toxic_comment(comment):
    """
    Detects whether a comment is toxic or not using a pre-trained model from Hugging Face Transformers.

    Args:
        comment (str): The comment to be classified.

    Returns:
        dict: A dictionary containing the classification results.
    """
    toxic_comment_detector = pipeline(model='martin-ha/toxic-comment-model')
    toxicity_classification = toxic_comment_detector(comment)
    return toxicity_classification

# test_function_code --------------------

def test_detect_toxic_comment():
    """
    Tests the detect_toxic_comment function with some test cases.
    """
    # Test case 1: A non-toxic comment
    comment1 = 'This is a great product!'
    classification1 = detect_toxic_comment(comment1)
    assert isinstance(classification1, dict), 'The result should be a dictionary.'

    # Test case 2: A toxic comment
    comment2 = 'You are a loser!'
    classification2 = detect_toxic_comment(comment2)
    assert isinstance(classification2, dict), 'The result should be a dictionary.'

    # Test case 3: An empty comment
    comment3 = ''
    classification3 = detect_toxic_comment(comment3)
    assert isinstance(classification3, dict), 'The result should be a dictionary.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_toxic_comment()