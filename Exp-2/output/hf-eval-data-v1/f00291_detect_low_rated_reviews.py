from transformers import pipeline


def detect_low_rated_reviews(review_text):
    """
    This function uses the Hugging Face Transformers library to analyze a product review and predict its sentiment.
    The sentiment is predicted in terms of star ratings (ranging from 1 to 5 stars), where 1 indicates a very negative sentiment, and 5 indicates a very positive sentiment.
    Low-rated product reviews are detected by filtering reviews with low-star ratings.
    
    Parameters:
    review_text (str): The product review text to analyze.
    
    Returns:
    str: A message indicating whether the review is low-rated or not.
    """
    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    result = sentiment_pipeline(review_text)
    if int(result[0]['label'][-1]) < 3: # Consider reviews with less than 3 stars negative
        return 'Low-rated product review detected'
    else:
        return 'Review is not low-rated'