# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_sentence_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate the similarity between two sentences using SentenceTransformer.

    Args:
        sentence1 (str): The first sentence to compare.
        sentence2 (str): The second sentence to compare.

    Returns:
        float: The similarity score between the two sentences. The score is between -1 and 1, where 1 means the sentences are identical, and -1 means they are completely different.
    """
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    sentences = [sentence1, sentence2]
    embeddings = model.encode(sentences)
    similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
    return similarity

# test_function_code --------------------

def test_calculate_sentence_similarity():
    """
    Test the calculate_sentence_similarity function.
    """
    sentence1 = 'I love going to the park'
    sentence2 = 'My favorite activity is visiting the park'
    similarity = calculate_sentence_similarity(sentence1, sentence2)
    assert 0.7 <= similarity <= 1.0, 'The sentences are expected to be similar.'

# call_test_function_code --------------------

test_calculate_sentence_similarity()