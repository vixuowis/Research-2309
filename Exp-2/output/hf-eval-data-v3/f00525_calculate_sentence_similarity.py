# function_import --------------------

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_sentence_similarity(sentences):
    """
    Calculate the similarity scores between a list of sentences using SentenceTransformer.

    Args:
        sentences (list): A list of sentences to calculate similarity scores.

    Returns:
        similarity_scores (numpy.ndarray): A 2D array containing the similarity scores between each pair of sentences.
    """
    # Load the pre-trained model
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Encode the sentences and calculate similarity scores
    embeddings = model.encode(sentences)
    similarity_scores = cosine_similarity(embeddings)

    return similarity_scores

# test_function_code --------------------

def test_calculate_sentence_similarity():
    """
    Test the function calculate_sentence_similarity.
    """
    sentences = ["This is an example sentence.", "Each sentence is converted.", "Calculate the similarity between sentences."]
    similarity_scores = calculate_sentence_similarity(sentences)
    assert similarity_scores.shape == (len(sentences), len(sentences)), 'The shape of similarity scores is not correct.'
    assert similarity_scores.dtype == np.float32, 'The data type of similarity scores is not correct.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_calculate_sentence_similarity()