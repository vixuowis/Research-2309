# requirements_file --------------------

!pip install -U numpy sentence-transformers

# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def find_similar_documents(query, documents):
    """
    Given a query sentence and a list of document sentences, find the documents that are
    semantically similar to the query.

    Args:
        query (str): The query sentence for which to find similar documents.
        documents (list): A list of sentences representing the documents.

    Returns:
        list: A list containing tuples of (document, similarity_score).
    """
    # Initialize the model
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    # Convert the query and documents to embeddings
    query_embedding = model.encode([query])[0]
    document_embeddings = model.encode(documents)
    
    # Calculate similarity (dot product works since vectors are normalized)
    similarity_scores = [(doc, np.dot(query_embedding, doc_emb))
                        for doc, doc_emb in zip(documents, document_embeddings)]
    
    # Sort the documents based on similarity scores
    sorted_docs = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    return sorted_docs

# test_function_code --------------------

def test_find_similar_documents():
    print("Testing started.")
    # Example query and documents
    query = 'An example query sentence'
    documents = ['This is an example document',
                 'Totally irrelevant sentence',
                 'A document similar to the query',
                 'Another relevant document example']

    # Expected: the two relevant documents should have higher scores
    print("Testing single query case.")
    results = find_similar_documents(query, documents)
    assert results[0][0].startswith('A document') and results[1][0].startswith('Another'), "Test failed: the most similar documents should be at the top."
    print("Single query test passed.")

    print("Testing finished.")

# Run tests
if __name__ == '__main__':
    test_find_similar_documents()