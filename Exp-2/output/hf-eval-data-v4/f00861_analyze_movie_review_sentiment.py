# requirements_file --------------------

!pip install -U transformers pytorch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_movie_review_sentiment(review_text):
    # Load sentiment-analysis pipeline with the specified model from Hugging Face Transformers
    sentiment_classifier = pipeline('sentiment-analysis', model='lvwerra/distilbert-imdb')
    
    # Invoke the classifier to predict the sentiment of the movie review
    sentiment_prediction = sentiment_classifier(review_text)
    
    # Return the sentiment prediction
    return sentiment_prediction

# test_function_code --------------------

def test_analyze_movie_review_sentiment():
    print("Testing started.")
    # Positive review test case
    positive_review = "This movie was fantastic! Great acting and story."
    positive_result = analyze_movie_review_sentiment(positive_review)
    assert positive_result[0]['label'] == 'POSITIVE', f"Test case failed: expected POSITIVE, got {positive_result[0]['label']}"

    # Negative review test case
    negative_review = "Terrible movie. I did not enjoy it at all."
    negative_result = analyze_movie_review_sentiment(negative_review)
    assert negative_result[0]['label'] == 'NEGATIVE', f"Test case failed: expected NEGATIVE, got {negative_result[0]['label']}"

    # Neutral review test case to showcase a potential boundary condition (not covered by the model)
    neutral_review = "The movie was average, nothing special."
    neutral_result = analyze_movie_review_sentiment(neutral_review)
    print(f"Neutral case result: {neutral_result}")
    print("Testing finished.")

# Run the test function
test_analyze_movie_review_sentiment()