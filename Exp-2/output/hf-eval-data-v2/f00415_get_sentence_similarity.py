# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def get_sentence_similarity(sentences):
    """
    This function calculates the similarity between sentences using SentenceTransformer.

    Args:
        sentences (list): A list of sentences for which the similarity is to be calculated.

    Returns:
        similarity_matrix (numpy.ndarray): A matrix containing the cosine similarity between each pair of sentences.
    """
    model = SentenceTransformer('nikcheerla/nooks-amd-detection-v2-full')
    embeddings = model.encode(sentences)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

# test_function_code --------------------

def test_get_sentence_similarity():
    """
    This function tests the get_sentence_similarity function by comparing the output with expected results.
    """
    sentences = ['This is an example sentence', 'Each sentence is converted']
    similarity_matrix = get_sentence_similarity(sentences)
    assert similarity_matrix.shape == (len(sentences), len(sentences))

# call_test_function_code --------------------

test_get_sentence_similarity()