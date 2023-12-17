# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_emotion_in_movie_review(text):
    """
    Identify the emotion expressed in a movie review using sentiment analysis.

    Parameters:
    text (str): A string containing the movie review to be analyzed.

    Returns:
    dict: The emotion classification result predicted by the model.
    """
    # Initialize the sentiment analysis model
    classifier = pipeline('sentiment-analysis', model='michellejieli/emotion_text_classifier')

    # Predict the emotion in the provided text
    result = classifier(text)

    return result

# test_function_code --------------------

def test_classify_emotion_in_movie_review():
    print("Testing classify_emotion_in_movie_review function.")
    # Test case 1: Check the function returns a non-empty result
    print("Test case [1/1] - Non-empty result")
    sample_review = "What a fantastic movie! It was so captivating."
    result = classify_emotion_in_movie_review(sample_review)
    assert result, f"Test case [1/1] failed: Expected a non-empty result, got {result}"
    print("Test case [1/1] passed!")

    print("Testing finished.")

# Run the test function
test_classify_emotion_in_movie_review()