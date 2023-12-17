# requirements_file --------------------

pip install -U sentence-transformers numpy scikit-learn

# function_import --------------------

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def find_similar_sentences(sentences, model_name='sentence-transformers/all-MiniLM-L12-v2'):
    """
    Find the most similar sentences in a list using a pre-trained SentenceTransformer model.

    Args:
        sentences: A list of sentences to compare for similarity.
        model_name: The name of the pre-trained SentenceTransformer model to use.

    Returns:
        A list of tuples containing sentence pairs with their similarity scores.

    Raises:
        ValueError: If the sentences list is empty.
    """
    if not sentences:
        raise ValueError("The list of sentences provided is empty.")

    # Load the pre-trained SentenceTransformer model
    model = SentenceTransformer(model_name)

    # Represent each sentence as a dense vector
    embeddings = model.encode(sentences)

    # Calculate cosine similarity between sentence embeddings
    similarity_matrix = cosine_similarity(embeddings)

    # Find the most similar sentence pairs
    similar_sentences = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similar_sentences.append(((sentences[i], sentences[j]), similarity_matrix[i][j]))

    # Sort by similarity score in descending order
    similar_sentences.sort(key=lambda x: x[1], reverse=True)
    return similar_sentences

# test_function_code --------------------

def test_find_similar_sentences():
    print("Testing started.")
    sentences = ['This is an example sentence.', 'Each sentence is converted.', 'This is another similar sentence.', 'No similarity here.']

    # Test case 1: Non-empty sentences list
    print("Testing case [1/3] started.")
    similar_sentences = find_similar_sentences(sentences)
    assert len(similar_sentences) > 0, f"Test case [1/3] failed: Expected non-empty result, got {similar_sentences}"

    # Test case 2: Check similarity order
    print("Testing case [2/3] started.")
    assert similar_sentences[0][1] >= similar_sentences[-1][1], f"Test case [2/3] failed: Expected most similar sentences at the start"

    # Test case 3: Check for ValueError when input is empty
    print("Testing case [3/3] started.")
    try:
        find_similar_sentences([])
        assert False, "Test case [3/3] failed: Expected ValueError for empty sentences list"
    except ValueError:
        pass
    print("Testing finished.")

# call_test_function_line --------------------

test_find_similar_sentences()