# requirements_file --------------------

import subprocess

requirements = ["transformers", "scikit-learn"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_review_similarity(review1, review2):
    """
    Calculate the cosine similarity between two book reviews.

    Args:
        review1 (str): The first book review.
        review2 (str): The second book review.

    Returns:
        float: The cosine similarity score between the review embeddings.

    Raises:
        ValueError: If the reviews are not strings or are empty.
    """
    if not isinstance(review1, str) or not isinstance(review2, str):
        raise ValueError('The reviews must be strings.')
    if not review1 or not review2:
        raise ValueError('The reviews cannot be empty.')

    tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')
    model = AutoModel.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')
    input_tensors = tokenizer([review1, review2], return_tensors='pt', padding=True, truncation=True)
    embeddings = model(**input_tensors).pooler_output
    similarity_score = cosine_similarity(embeddings[0].detach().numpy().reshape(1, -1), embeddings[1].detach().numpy().reshape(1, -1))[0][0]
    return similarity_score

# test_function_code --------------------

def test_calculate_review_similarity():
    print("Testing started.")
    # Testing the normal case with two non-empty strings
    review1 = "This is a review of a book."
    review2 = "This is another review of the book."
    print("Testing case [1/3] started.")
    similarity_score = calculate_review_similarity(review1, review2)
    assert isinstance(similarity_score, float), "Test case [1/3] failed: The function should return a float similarity score."
    # Testing with an empty string as review1
    try:
        print("Testing case [2/3] started.")
        calculate_review_similarity('', review2)
        assert False, "Test case [2/3] failed: The function should raise a ValueError for empty strings."
    except ValueError:
        pass    # Test passed
    # Testing with integer inputs instead of strings
    try:
        print("Testing case [3/3] started.")
        calculate_review_similarity(123, 456)
        assert False, "Test case [3/3] failed: The function should raise a ValueError for non-string inputs."
    except ValueError:
        pass  # Test passed
    print("Testing finished.")

# call_test_function_line --------------------

test_calculate_review_similarity()