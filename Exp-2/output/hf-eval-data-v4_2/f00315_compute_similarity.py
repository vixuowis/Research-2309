# requirements_file --------------------

pip install -U sentence-transformers sklearn

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def compute_similarity(documents): 
    """
    Computes the cosine similarity matrix between given documents.

    Args:
        documents (List[str]): A list of text documents to be compared.

    Returns:
        numpy.ndarray: A matrix of similarity scores between documents.

    Raises:
        ValueError: If the input is not a non-empty list of strings.
    """
    if not isinstance(documents, list) or not documents or not all(isinstance(doc, str) for doc in documents):
        raise ValueError('Input must be a non-empty list of strings.')

    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(documents)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

# test_function_code --------------------

def test_compute_similarity():
    print("Testing started.")

    # Testing case 1: Comparing identical sentences
    print("Testing case [1/2] started.")
    identical_documents = ["This is a document.", "This is a document."]
    similarity_matrix = compute_similarity(identical_documents)
    assert (similarity_matrix[0][1] == 1), f"Test case [1/2] failed: Expected similarity of 1, got {similarity_matrix[0][1]}"

    # Testing case 2: Comparing different sentences
    print("Testing case [2/2] started.")
    different_documents = ["This is the first document.", "This is the second document."]
    similarity_matrix = compute_similarity(different_documents)
    assert (similarity_matrix[0][1] < 1), f"Test case [2/2] failed: Expected similarity less than 1, got {similarity_matrix[0][1]}"
    print("Testing finished.")

# call_test_function_line --------------------

test_compute_similarity()