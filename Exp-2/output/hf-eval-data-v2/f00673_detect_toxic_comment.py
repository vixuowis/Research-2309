# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_toxic_comment(comment):
    """
    Detects whether a comment is toxic or not.

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
    Tests the detect_toxic_comment function with a sample comment.
    """
    comment = 'This is a test comment.'
    classification = detect_toxic_comment(comment)
    assert isinstance(classification, dict), 'The result should be a dictionary.'
    assert 'label' in classification, 'The result should have a label.'
    assert 'score' in classification, 'The result should have a score.'

# call_test_function_code --------------------

test_detect_toxic_comment()