from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def sentence_similarity(sentences):
    """
    This function takes a list of sentences and returns a similarity matrix.
    The similarity is calculated using the SentenceTransformer model from Hugging Face Transformers.
    Each sentence is converted into a 384 dimensional vector and the cosine similarity between these vectors is calculated.
    
    Args:
    sentences (list): A list of sentences for which the similarity is to be calculated.
    
    Returns:
    np.array: A similarity matrix where each element [i,j] represents the similarity between sentence i and sentence j.
    """
    # Load the pre-trained model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    
    # Encode the sentences to get their embeddings
    embeddings = model.encode(sentences)
    
    # Calculate the similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    return similarity_matrix