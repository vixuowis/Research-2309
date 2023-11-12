# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_sentence_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate the similarity between two sentences using SentenceTransformer.

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
    assert abs(calculate_sentence_similarity('This is a test sentence', 'This is a test sentence') - 1.0) < 0.01
    assert abs(calculate_sentence_similarity('This is a test sentence', 'This is another test sentence') - 0.8) < 0.1
    assert abs(calculate_sentence_similarity('This is a test sentence', 'Completely different sentence') - 0.5) < 0.1
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_calculate_sentence_similarity())