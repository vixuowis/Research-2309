# requirements_file --------------------

!pip install -U sentence-transformers sklearn

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def find_most_similar_sentence_pairs(sentences):
    """
    Find the most similar sentence pairs in a list of sentences using
    SentenceTransformer.

    Args:
    sentences (list of str): A list of sentence strings for similarity comparison.

    Returns:
    list of tuple: A list containing pairs of sentences and their similarity scores.
    """
    # Initialize the SentenceTransformer model
    model = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')

    # Encode the sentences
    embeddings = model.encode(sentences)

    # Compute the cosine similarity between sentence embeddings
    similarity_matrix = cosine_similarity(embeddings)

    # Find and return the most similar sentence pairs
    similar_pairs = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similar_pairs.append(((sentences[i], sentences[j]), similarity_matrix[i][j]))

    # Sort by similarity score in descending order
    similar_pairs.sort(key=lambda pair: pair[1], reverse=True)

    return similar_pairs


# test_function_code --------------------

def test_find_most_similar_sentence_pairs():
    print("Testing find_most_similar_sentence_pairs function.")

    # Test data
    sentences = [
        "I have a dog",
        "My dog loves to play",
        "There is a cat in our house",
        "The cat and the dog get along well"
    ]

    # Expected similar pairs
    expected_pairs = [
        ("My dog loves to play", "The cat and the dog get along well"),
        ("I have a dog", "My dog loves to play")
    ]

    # Running the test
    result_pairs = find_most_similar_sentence_pairs(sentences)

    # Checking if the top 2 matches are in the expected_pairs
    assert all(pair in expected_pairs for pair in result_pairs[:2]), "Test failed. The most similar sentence pairs do not match the expected pairs."

    print("Testing completed successfully.")

# Run the test function
test_find_most_similar_sentence_pairs()
