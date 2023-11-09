from transformers import pipeline

# This function analyzes the sentiment of a product review using the Hugging Face Transformers library.
# It uses the 'nlptown/bert-base-multilingual-uncased-sentiment' model, which is trained for sentiment analysis on product reviews in six languages: English, Dutch, German, French, Spanish, and Italian.
# The function takes a string of text (the review) as input and returns the sentiment score as a number of stars (between 1 and 5).
# High star ratings indicate a positive sentiment, while low star ratings indicate negative sentiment.
def analyze_review_sentiment(review_text):
    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    review_sentiment = sentiment_pipeline(review_text)
    return review_sentiment