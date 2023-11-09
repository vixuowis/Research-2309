from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


def calculate_sentence_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate the similarity between two sentences using SentenceTransformer model.

    Args:
        sentence1 (str): The first sentence to compare.
        sentence2 (str): The second sentence to compare.

    Returns:
        float: The similarity score between the two sentences. A high score indicates that the sentences are semantically similar.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    sentence1_embedding = model.encode(sentence1)
    sentence2_embedding = model.encode(sentence2)
    similarity = 1 - cosine(sentence1_embedding, sentence2_embedding)
    return similarity