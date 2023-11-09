from sentence_transformers import CrossEncoder

def classify_news_articles(news_article):
    '''
    This function classifies a given news article into one of the specified categories: 'technology', 'sports', or 'politics'.
    It uses the CrossEncoder model from the sentence_transformers library, which is trained on a large dataset of natural language inference tasks.
    This model is effective for zero-shot classification, which can classify text into specified categories without labeled training data.
    
    Parameters:
    news_article (str): The content of the news article to be classified.
    
    Returns:
    str: The category that the news article most likely belongs to.
    '''
    cross_encoder = CrossEncoder('cross-encoder/nli-roberta-base')
    candidate_labels = ['technology', 'sports', 'politics']
    scores = cross_encoder.predict([{'sentence1': news_article, 'sentence2': label} for label in candidate_labels])
    return candidate_labels[scores.index(max(scores))]