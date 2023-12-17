# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_toxic_comment(comment):
    """
    Function to assess if a user-generated comment is toxic.
    
    Parameters:
    comment (str): The comment text to be assessed.

    Returns:
    dict: A dictionary with the comment and its toxicity score.
    """
    # Initialize the classifier using the pre-trained model
    toxic_classifier = pipeline(model='martin-ha/toxic-comment-model')
    
    # Evaluate the toxicity score for the comment
    toxicity_score = toxic_classifier(comment)
    
    # Return the result as a dictionary
    return {'comment': comment, 'toxicity_score': toxicity_score[0]}

# test_function_code --------------------

def test_classify_toxic_comment():
    print("Testing started.")
    # Sample comments for testing
    comments = [
        "I love the effort you've put in, great job!",
        "You are an idiot and your idea is stupid.",
        "This is a user-generated comment."
    ]

    for i, comment in enumerate(comments):
        print(f"Testing case [{i+1}/{len(comments)}] started.")
        result = classify_toxic_comment(comment)
        
        assert 'label' in result['toxicity_score'], f"Test case [{i+1}/{len(comments)}] failed: Missing label."
        assert 'score' in result['toxicity_score'], f"Test case [{i+1}/{len(comments)}] failed: Missing score."
        
        print(f"Test case [{i+1}/{len(comments)}] passed.")

    print("Testing finished.")

# Run the test function
test_classify_toxic_comment()