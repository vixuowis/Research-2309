from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_sentence_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate the similarity between two sentences using SentenceTransformer.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        float: The similarity score between the two sentences. The score is between -1 and 1, where 1 means the sentences are identical.

    Raises:
        ValueError: If the input sentences are not strings.
    """
    if not isinstance(sentence1, str) or not isinstance(sentence2, str):
        raise ValueError('Both inputs should be strings.')

    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    sentences = [sentence1, sentence2]
    embeddings = model.encode(sentences)
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity