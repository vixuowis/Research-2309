# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_movie_review(review_text):
    # Initialize the zero-shot classification pipeline.
    nlp = pipeline('zero-shot-classification', model='valhalla/distilbart-mnli-12-6')
    # Define the candidate labels
    candidate_labels = ['positive', 'negative']
    # Perform zero-shot classification to identify user opinion
    result = nlp(review_text, candidate_labels)
    # Return the classification results
    return result

# test_function_code --------------------

def test_analyze_movie_review():
    print("Testing analyze_movie_review function.")
    # Test with a positive review
    positive_review = "Inception is a masterpiece, with thrilling visuals and an outstanding story."
    result_positive = analyze_movie_review(positive_review)
    assert result_positive['labels'][0] == 'positive', f"Test failed: Positive review not classified correctly."

    # Test with a negative review
    negative_review = "Inception is overrated and confusing."
    result_negative = analyze_movie_review(negative_review)
    assert result_negative['labels'][0] == 'negative', f"Test failed: Negative review not classified correctly."

    # Test with a neutral review
    neutral_review = "Inception is a film I watched."
    result_neutral = analyze_movie_review(neutral_review)
    assert result_neutral['labels'][0] in ['positive', 'negative'], f"Test failed: Neutral review not classified correctly."

    print("All tests passed!")

# Run the test
test_analyze_movie_review()