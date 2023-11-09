from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_article_similarity(breaking_news_text, other_article_texts):
    '''
    This function calculates the similarity between the main text of a breaking news article and other articles.
    It uses the SentenceTransformer model from Hugging Face Transformers to encode the texts into a 512-dimensional dense vector space.
    The similarity between the embeddings of each article is then calculated using cosine similarity.
    
    Parameters:
    breaking_news_text (str): The main text of the breaking news article.
    other_article_texts (list): A list of texts from other articles.
    
    Returns:
    breaking_news_similarities (list): A list of similarity scores between the breaking news article and other articles.
    '''
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    texts = [breaking_news_text] + other_article_texts
    embeddings = model.encode(texts)
    similarity_matrix = cosine_similarity(embeddings)
    breaking_news_similarities = similarity_matrix[0, 1:]
    return breaking_news_similarities