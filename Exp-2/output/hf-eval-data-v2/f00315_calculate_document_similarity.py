# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_document_similarity(documents):
    """
    This function calculates the similarity between given documents using SentenceTransformer.

    Args:
        documents (list): A list of documents to compare.

    Returns:
        similarity_matrix (numpy.ndarray): A matrix containing the cosine similarity between each pair of documents.
    """
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(documents)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

# test_function_code --------------------

def test_calculate_document_similarity():
    """
    This function tests the calculate_document_similarity function.
    It uses a small set of documents and checks if the similarity matrix is correctly computed.
    """
    documents = ['This is a test document.', 'This is another test document.', 'This is yet another test document.']
    similarity_matrix = calculate_document_similarity(documents)
    assert similarity_matrix.shape == (len(documents), len(documents)), 'The shape of the similarity matrix is incorrect.'
    assert (similarity_matrix >= 0).all() and (similarity_matrix <= 1).all(), 'The similarity matrix should contain values between 0 and 1.'

# call_test_function_code --------------------

test_calculate_document_similarity()