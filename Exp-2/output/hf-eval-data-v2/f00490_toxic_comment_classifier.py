# function_import --------------------

from transformers import pipeline

# function_code --------------------

def toxic_comment_classifier(comment):
    """
    This function classifies a comment as toxic or not using a pre-trained model from Hugging Face Transformers.

    Args:
        comment (str): The comment to be classified.

    Returns:
        float: The toxicity score of the comment. A higher score indicates a higher likelihood of the comment being toxic.
    """
    toxic_classifier = pipeline(model='martin-ha/toxic-comment-model')
    toxicity_score = toxic_classifier(comment)
    return toxicity_score[0]['label'], toxicity_score[0]['score']

# test_function_code --------------------

def test_toxic_comment_classifier():
    """
    This function tests the toxic_comment_classifier function with a sample comment.
    """
    comment = 'This is a user-generated comment.'
    label, score = toxic_comment_classifier(comment)
    assert label in ['TOXIC', 'NOT_TOXIC'], 'Invalid label returned'
    assert 0 <= score <= 1, 'Invalid score returned'

# call_test_function_code --------------------

test_toxic_comment_classifier()