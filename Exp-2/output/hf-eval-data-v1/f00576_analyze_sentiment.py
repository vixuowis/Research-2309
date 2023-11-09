from transformers import pipeline

# Function to analyze sentiment of a text
# Uses the 'finiteautomata/beto-sentiment-analysis' model from the Hugging Face Transformers library
# This model is trained on the TASS 2020 corpus and can analyze the sentiment of text in Spanish
# The function takes a string as input and returns a sentiment analysis result

def analyze_sentiment(review_text):
    # Create a sentiment analysis model
    sentiment_model = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    # Use the model to analyze the sentiment of the review text
    sentiment_result = sentiment_model(review_text)
    # Return the sentiment analysis result
    return sentiment_result