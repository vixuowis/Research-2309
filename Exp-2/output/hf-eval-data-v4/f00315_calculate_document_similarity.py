# requirements_file --------------------

!pip install -U sentence-transformers sklearn

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_document_similarity(documents):
    """
    Calculate the similarity between multiple documents using the SentenceTransformer model.

    Args:
        documents (list[str]): A list of documents (strings) to be compared.

    Returns:
        numpy.ndarray: A matrix of cosine similarity scores between the documents.
    """
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(documents)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

# test_function_code --------------------

def test_calculate_document_similarity():
    print("Testing started.")
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A fast, dark-colored fox leaps over the sleepy canine",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit"
    ]

    similarity_matrix = calculate_document_similarity(documents)
    assert similarity_matrix.shape == (3, 3), f"Test case failed: Expected similarity matrix to have shape (3, 3), got {similarity_matrix.shape}"
    print("Similarity matrix:\n", similarity_matrix)
    print("All test cases passed!")

# Running the test function
test_calculate_document_similarity()