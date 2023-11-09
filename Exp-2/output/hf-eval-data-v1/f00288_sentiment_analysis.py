from transformers import pipeline

# This function is used to perform sentiment analysis on a given text.
# It uses the 'pipeline' function from transformers to load the sentiment analysis model.
# The model used is 'Seethal/sentiment_analysis_generic_dataset', which has been fine-tuned on a classified dataset for text classification.
# The function takes a string as input and returns the sentiment of the text as output.
# The sentiment can be either 'positive' or 'negative'.
def sentiment_analysis(review):
    sentiment_analysis = pipeline('text-classification', model='Seethal/sentiment_analysis_generic_dataset')
    result = sentiment_analysis(review)
    return result