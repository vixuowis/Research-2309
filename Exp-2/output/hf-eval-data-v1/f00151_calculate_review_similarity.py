from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_review_similarity(restaurant_reviews):
    """
    This function calculates the similarity scores for different restaurant reviews.
    It uses the SentenceTransformer model from Hugging Face Transformers to convert each review into a vector that captures the review's semantic information.
    It then calculates the cosine similarity between each pair of review vectors to obtain similarity scores.
    
    Parameters:
    restaurant_reviews (list): A list of restaurant reviews.
    
    Returns:
    similarity_matrix (numpy.ndarray): A matrix of similarity scores.
    """
    # Load the SentenceTransformer model
    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
    
    # Encode each restaurant review to get sentence embeddings
    review_embeddings = model.encode(restaurant_reviews)
    
    # Calculate cosine similarity between each pair of review embeddings
    similarity_matrix = cosine_similarity(review_embeddings)
    
    return similarity_matrix