# requirements_file --------------------

import subprocess

requirements = ["sentence-transformers", "scikit-learn"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_sentence_similarity(sentence_1: str, sentence_2: str) -> float:
    """
    Calculate the similarity between two given sentences using embeddings
    generated by the 'sentence-transformers/paraphrase-MiniLM-L6-v2' model.

    Args:
        sentence_1: The first sentence to compare.
        sentence_2: The second sentence to compare.

    Returns:
        A float representing the cosine similarity between the sentence embeddings.
    """
    sentences = [sentence_1, sentence_2]
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
    return similarity

# test_function_code --------------------

def test_calculate_sentence_similarity():
    print("Testing started.")

    sentence_pairs = [
        ("I have a pen.", "I have an apple."),
        ("Birds fly in the sky.", "Planes fly in the sky."),
        ("A cat is a pet.", "A sofa is a piece of furniture.")
    ]
    expected_similarity_scores = [0.5, 0.9, 0.1]  # hypothetical similarity scores for test cases

    for i, (sentences, expected) in enumerate(zip(sentence_pairs, expected_similarity_scores), start=1):
        print(f"Testing case [{i}/{len(sentence_pairs)}] started.")
        result = calculate_sentence_similarity(*sentences)
        assert abs(result - expected) < 0.2, f"Test case [{i}/{len(sentence_pairs)}] failed: Expected similarity around {expected}, but got {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_calculate_sentence_similarity()