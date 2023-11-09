from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans


def group_articles_by_topic(sentences):
    """
    This function groups articles discussing the specific topic using SentenceTransformer.
    It maps sentences & paragraphs to a 512 dimensional dense vector space and can be used for tasks like clustering or semantic search.
    
    Args:
    sentences (list): A list of sentences from the articles.
    
    Returns:
    list: A list of cluster labels for the sentences.
    """
    # Load the SentenceTransformer model
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
    
    # Encode the sentences to get their embeddings
    embeddings = model.encode(sentences)
    
    # Use KMeans clustering to group the sentences
    kmeans = KMeans(n_clusters=5, random_state=0).fit(embeddings)
    
    return kmeans.labels_