# function_import --------------------

from transformers import pipeline

# function_code --------------------

def toxic_comment_classifier(comment):
    """
    Classify a comment as toxic or not using a pre-trained model from Hugging Face Transformers.

    Args:
        comment (str): The comment to be classified.

    Returns:
        str: The classification result, either 'TOXIC' or 'NOT_TOXIC'.
    """
    toxic_classifier = pipeline(model='martin-ha/toxic-comment-model')
    toxicity_score = toxic_classifier(comment)
    return 'TOXIC' if toxicity_score[0]['label'] == 'LABEL_0' else 'NOT_TOXIC'

# test_function_code --------------------

def test_toxic_comment_classifier():
    """
    Test the toxic_comment_classifier function.
    """
    assert toxic_comment_classifier('This is a user-generated comment.') == 'NOT_TOXIC', 'Test Case 1 Failed'
    assert toxic_comment_classifier('You are stupid.') == 'TOXIC', 'Test Case 2 Failed'
    assert toxic_comment_classifier('Have a nice day!') == 'NOT_TOXIC', 'Test Case 3 Failed'
    print('All Tests Passed')

# call_test_function_code --------------------

test_toxic_comment_classifier()