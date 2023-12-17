# requirements_file --------------------

!pip install -U numpy sentence-transformers

# function_import --------------------

import numpy as np
from sentence_transformers import SentenceTransformer

# function_code --------------------

def are_questions_similar(question1, question2, model):
    """
    Determine if two questions are similar using a SentenceTransformer model.

    Args:
    question1 (str): The first question to compare.
    question2 (str): The second question to compare.
    model (SentenceTransformer): Pre-trained transformer model for calculating sentence embeddings.

    Returns:
    float: The cosine similarity between the two questions.
    """
    # Encode the questions to get their embeddings
    embedding1 = model.encode(question1)
    embedding2 = model.encode(question2)

    # Compute cosine similarity
    similarity = np.inner(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity

# test_function_code --------------------

def test_are_questions_similar():
    print("Testing are_questions_similar function.")

    # Load the model
    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')

    # Test cases
    test_cases = [
        ('What time is it?', 'Can you tell me the current time?'),
        ('How is the weather today?', 'Is it raining outside?'),
        ('Do you have any pets?', 'What pets do you have?')
    ]

    # Expected similarity is high for questions 1, low for question 2 and moderate for question 3
    expected_results = [0.7, 0.3, 0.5]

    # Test each case
    for i, (q1, q2) in enumerate(test_cases):
        similarity = are_questions_similar(q1, q2, model)
        assert similarity > expected_results[i], f"Test case [{i+1}] failed: Expected similarity > {expected_results[i]}, got {similarity}."

    print("All tests passed.")

# Run the test
print("Running tests...")
test_are_questions_similar()