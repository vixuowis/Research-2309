# requirements_file --------------------

!pip install -U sentence-transformers numpy

# function_import --------------------

from sentence_transformers import SentenceTransformer
import numpy as np

# function_code --------------------

def analyze_character_similarity(conversations):
    """
    Given a list of conversations (each conversation is a list of sentences),
    this function computes the similarity between each pair of conversations
    using sentence embeddings.

    :param conversations: A list of conversations (list of list of strings)
    :return: A numpy array of similarity scores between conversations
    """
    # Initialize the SentenceTransformer model
    model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

    # Compute embeddings for each conversation
    conversation_embeddings = []
    for conversation in conversations:
        embedding = model.encode(conversation)
        conversation_embeddings.append(np.mean(embedding, axis=0))

    # Calculate cosine similarity between conversation embeddings
    similarity_matrix = np.inner(conversation_embeddings, conversation_embeddings)
    return similarity_matrix

# test_function_code --------------------

def test_analyze_character_similarity():
    print("Testing started.")
    # Sample conversations for testing
    conversations = [
        ["How are you today?", "I'm fine, thank you."],
        ["How do you do?", "I am well, thanks."]]

    similarity_matrix = analyze_character_similarity(conversations)

    # Test case 1: The return type is a numpy array
    assert isinstance(similarity_matrix, np.ndarray), "Test case failed: The result should be a numpy array."

    # Test case 2: The similarity matrix is square with dimensions equal to the number of conversations
    assert similarity_matrix.shape == (len(conversations), len(conversations)), "Test case failed: The similarity matrix should be square."

    # Test case 3: Diagonal elements of the similarity matrix are 1 (self-similarity)
    assert np.allclose(np.diag(similarity_matrix), 1), "Test case failed: The diagonal elements should be 1 (self-similarity)."

    print("Testing finished.")

# Run the test
print("We are starting the tests:")
print("--------------------------")
test_analyze_character_similarity()