# requirements_file --------------------

!pip install -U sentence_transformers transformers

# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def classify_news_article(news_article, candidate_labels):
    """
    Classify a given news article into one of the categories: technology, sports, or politics.

    Parameters:
        news_article (str): The content of the news article to classify.
        candidate_labels (list of str): The categories to classify the news article into.

    Returns:
        str: The category with the highest score as predicted by the zero-shot classification model.
    """
    # Initialize the CrossEncoder model
    cross_encoder = CrossEncoder('cross-encoder/nli-roberta-base')

    # Prepare the input for the CrossEncoder
    input_data = [{ 'sentence1': news_article, 'sentence2': label } for label in candidate_labels]

    # Predict the scores for each category
    scores = cross_encoder.predict(input_data)

    # Find the index of the category with the highest score
    max_score_index = scores.argmax()

    # Return the category with the highest score
    return candidate_labels[max_score_index]

# test_function_code --------------------

def test_classify_news_article():
    print("Testing started.")
    # Example news article
    news_article = 'Apple just announced the newest iPhone X'
    candidate_labels = ['technology', 'sports', 'politics']

    # Expected category
    expected_category = 'technology'

    # Run the classification function
    predicted_category = classify_news_article(news_article, candidate_labels)

    # Check if the predicted category matches the expected category
    assert predicted_category == expected_category, f"Test case failed: expected {expected_category}, got {predicted_category}"

    print("Testing finished.")

# Run the test function
test_classify_news_article()