# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_sentence_similarity(sentence1: str, sentence2: str) -> float:
    """
    This function calculates the similarity between two sentences using the SentenceTransformer model.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        float: The similarity score between the two sentences.
    """
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    sentences = [sentence1, sentence2]
    embeddings = model.encode(sentences)
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity

# test_function_code --------------------

def test_calculate_sentence_similarity():
    """
    This function tests the calculate_sentence_similarity function.
    """
    sentence1 = 'This is an example sentence'
    sentence2 = 'Each sentence is converted'
    similarity = calculate_sentence_similarity(sentence1, sentence2)
    assert 0 <= similarity <= 1, 'The similarity score should be between 0 and 1'

# call_test_function_code --------------------

test_calculate_sentence_similarity()