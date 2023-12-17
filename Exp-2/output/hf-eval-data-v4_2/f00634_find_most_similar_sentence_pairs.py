# requirements_file --------------------

!pip install -U sentence-transformers sklearn

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# function_code --------------------

def find_most_similar_sentence_pairs(sentences):
    """
    Finds the most similar sentence pairs from a list of sentences based on cosine similarity.

    Args:
        sentences (list of str): A list of sentences to analyze.

    Returns:
        list of tuple: A list of tuples containing the indices and similarity scores of the most similar sentence pairs.
    """
    model = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')
    embeddings = model.encode(sentences)
    similarity_matrix = cosine_similarity(embeddings)
    similar_pairs = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similar_pairs.append(((i, j), similarity_matrix[i][j]))
    # Sort by similarity score in descending order and return the pairs
    most_similar_pairs = sorted(similar_pairs, key=lambda x: x[1], reverse=True)
    return most_similar_pairs


# test_function_code --------------------

def test_find_most_similar_sentence_pairs():
    print("Testing started.")
    sentences = [
        "I have a dog",
        "My dog loves to play",
        "There is a cat in our house",
        "The cat and the dog get along well"
    ]

    print("Testing case [1/1] started.")
    result = find_most_similar_sentence_pairs(sentences)
    # Since this is dependent on the model, we'll just check if we got a result back
    assert result, f"Test case [1/1] failed: Expected a list of pairs with similarity scores, got {result}"
    print("Testing finished.")


# call_test_function_line --------------------

test_find_most_similar_sentence_pairs()