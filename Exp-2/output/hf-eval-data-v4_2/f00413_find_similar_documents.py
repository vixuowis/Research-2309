# requirements_file --------------------

pip install -U sentence-transformers

# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def find_similar_documents(descriptions, repository):
    """
    Find documents in the repository that are semantically similar to the descriptions provided.

    Args:
        descriptions (List[str]): A list of sentences or paragraphs to find similar documents for.
        repository (List[str]): A list of documents to search within.

    Returns:
        List[Tuple[int, float]]: A list of tuples with the index of the similar document and the similarity score.

    Raises:
        ValueError: If 'descriptions' or 'repository' are not lists or are empty.
    """
    if not isinstance(descriptions, list) or not descriptions:
        raise ValueError("'descriptions' must be a non-empty list.")
    if not isinstance(repository, list) or not repository:
        raise ValueError("'repository' must be a non-empty list.")

    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    description_embeddings = model.encode(descriptions)
    repository_embeddings = model.encode(repository)

    # Assuming a function `calculate_similarity` exists to compare the embeddings
    # This can be cosine similarity, euclidean distance, or any other metric
    similar_documents = []
    for idx, rep_emb in enumerate(repository_embeddings):
        for desc_emb in description_embeddings:
            score = calculate_similarity(desc_emb, rep_emb)
            if score > threshold:
                similar_documents.append((idx, score))
    return similar_documents

# test_function_code --------------------

def test_find_similar_documents():
    print("Testing started.")
    test_descriptions = ['This document is about machine learning.', 'Natural language processing and its applications.']
    test_repository = ['Leveraging machine learning for better data analysis.',
                       'An intro to natural language processing.',
                       'Understanding deep learning techniques.']

    # Test case 1: Check if the function returns a list
    print("Testing case [1/3] started.")
    result = find_similar_documents(test_descriptions, test_repository)
    assert isinstance(result, list), "Test case [1/3] failed: The result is not a list."

    # Test case 2: Check if the function raises a ValueError for empty descriptions
    print("Testing case [2/3] started.")
    try:
        find_similar_documents([], test_repository)
        assert False, "Test case [2/3] failed: No ValueError raised for empty descriptions."
    except ValueError:
        pass

    # Test case 3: Check if the function raises a ValueError for empty repository
    print("Testing case [3/3] started.")
    try:
        find_similar_documents(test_descriptions, [])
        assert False, "Test case [3/3] failed: No ValueError raised for empty repository."
    except ValueError:
        pass
    print("Testing finished.")

# call_test_function_line --------------------

test_find_similar_documents()