from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_sentence_similarity(sentence1: str, sentence2: str) -> float:
    '''
    This function calculates the similarity between two sentences using the SentenceTransformer model.
    It uses the 'sentence-transformers/paraphrase-MiniLM-L6-v2' model from Hugging Face Transformers.
    The function returns a similarity score between -1 and 1.
    
    Parameters:
    sentence1 (str): The first sentence to compare.
    sentence2 (str): The second sentence to compare.
    
    Returns:
    float: The similarity score between the two sentences.
    '''
    sentences = [sentence1, sentence2]
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
    return similarity