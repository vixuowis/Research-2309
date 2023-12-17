# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_toxic_comments(comment: str) -> bool:
    """
    Classify a comment as toxic or non-toxic using a pre-trained DistilBERT model.

    Parameters:
    comment (str): The comment text to be classified.

    Returns:
    bool: True if the comment is toxic, False otherwise.
    """
    # Initialize the pre-trained model
    toxic_comment_detector = pipeline(model='martin-ha/toxic-comment-model')
    # Classify the comment
    result = toxic_comment_detector(comment)
    # Check if the comment is toxic
    return result[0]['label'] == 'LABEL_1', result[0]['score']


# test_function_code --------------------

def test_classify_toxic_comments():
    print("Testing classify_toxic_comments function...")

    # Sample comment
    comment_positive = "You are so stupid!"
    # Expected to be toxic
    toxic, confidence = classify_toxic_comments(comment_positive)
    assert toxic is True, f"Expected 'You are so stupid!' to be toxic, got: {toxic}"

    comment_negative = "Thank you for your help!"
    # Expected to be non-toxic
    toxic, confidence = classify_toxic_comments(comment_negative)
    assert toxic is False, f"Expected 'Thank you for your help!' to be non-toxic, got: {toxic}"

    print("All test cases passed successfully!")

# Run the test function
test_classify_toxic_comments()
