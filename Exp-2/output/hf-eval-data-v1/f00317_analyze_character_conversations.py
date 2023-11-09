from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def analyze_character_conversations(conversation_A, conversation_B):
    """
    This function analyzes how characters in a book are connected and if they share any similarity based on their conversation.
    It uses the SentenceTransformer model from Hugging Face Transformers to map sentences to high-dimensional dense vector spaces.
    The similarity between different characters' embeddings is then calculated to find connections or shared themes in their conversations.
    
    Parameters:
    conversation_A (list): List of sentences from character A
    conversation_B (list): List of sentences from character B
    
    Returns:
    float: Similarity score between the conversations of the two characters
    """
    
    # Create an instance of the SentenceTransformer class with the model 'sentence-transformers/all-roberta-large-v1'
    model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    
    # Encode the conversations using the model's encode method
    embeddings_A = model.encode(conversation_A)
    embeddings_B = model.encode(conversation_B)
    
    # Compute the mean of the embeddings for each conversation
    mean_A = np.mean(embeddings_A, axis=0)
    mean_B = np.mean(embeddings_B, axis=0)
    
    # Compute the cosine similarity between the mean embeddings of the two conversations
    similarity = cosine_similarity([mean_A], [mean_B])[0][0]
    
    return similarity