# requirements_file --------------------

!pip install -U sentence-transformers sklearn

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_sentence_similarity(sentences):
    """
    Calculate the cosine similarity scores between each pair of sentences.

    Args:
    sentences (list): A list of sentences to be compared for similarity.

    Returns:
    list: A list of lists containing similarity scores.
    """
    # Load the pre-trained model
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Encode the sentences to get their embeddings
    embeddings = model.encode(sentences)

    # Calculate the cosine similarity between sentence embeddings
    similarity_scores = cosine_similarity(embeddings)

    return similarity_scores

# test_function_code --------------------

def test_calculate_sentence_similarity():
    print("Testing calculate_sentence_similarity.")

    # Test case 1: Similar sentences
    sentences_similar = [
        "This is an example sentence.",
        "This sentence serves as an example."
    ]
    scores_similar = calculate_sentence_similarity(sentences_similar)
    assert scores_similar[0][1] > 0.7, f"Test case 1 failed: Expected high similarity, got {scores_similar[0][1]}"

    # Test case 2: Dissimilar sentences
    sentences_dissimilar = [
        "This is an example sentence.",
        "Completely unrelated text goes here."
    ]
    scores_dissimilar = calculate_sentence_similarity(sentences_dissimilar)
    assert scores_dissimilar[0][1] < 0.3, f"Test case 2 failed: Expected low similarity, got {scores_dissimilar[0][1]}"

    # Test case 3: Identical sentences
    sentences_identical = [
        "This is an example sentence.",
        "This is an example sentence."
    ]
    scores_identical = calculate_sentence_similarity(sentences_identical)
    assert scores_identical[0][1] == 1.0, f"Test case 3 failed: Expected identical similarity, got {scores_identical[0][1]}"

    print("All test cases passed for calculate_sentence_similarity.")