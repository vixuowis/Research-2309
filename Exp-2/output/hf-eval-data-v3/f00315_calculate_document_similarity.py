# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_document_similarity(documents):
    """
    Calculate the similarity between given documents using SentenceTransformer.

    Args:
        documents (list): A list of documents to calculate similarity.

    Returns:
        similarity_matrix (numpy.ndarray): A matrix containing the similarity scores between the documents.
    """
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(documents)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

# test_function_code --------------------

def test_calculate_document_similarity():
    documents = ["This is a document.", "This is another document.", "This is yet another document."]
    similarity_matrix = calculate_document_similarity(documents)
    assert similarity_matrix.shape == (3, 3)
    assert similarity_matrix[0, 0] == 1.0
    assert similarity_matrix[0, 1] != 1.0
    assert similarity_matrix[0, 2] != 1.0
    return 'All Tests Passed'

# call_test_function_code --------------------

test_calculate_document_similarity()