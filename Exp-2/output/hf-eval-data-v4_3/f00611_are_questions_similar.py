# requirements_file --------------------

import subprocess

requirements = ["sentence-transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from sentence_transformers import SentenceTransformer
import numpy as np

# function_code --------------------

def are_questions_similar(question1, question2):
    """
    Determines if two questions are semantically similar using SentenceTransformer model.

    Args:
        question1 (str): The first question to compare.
        question2 (str): The second question to compare.

    Returns:
        float: Semantic similarity between the two questions (ranging from -1 to 1).

    Raises:
        ValueError: If either question1 or question2 is not a valid string.
    """
    if not isinstance(question1, str) or not isinstance(question2, str):
        raise ValueError('Both questions must be strings.')

    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
    embedding1 = model.encode(question1)
    embedding2 = model.encode(question2)
    similarity = np.inner(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity

# test_function_code --------------------

def test_are_questions_similar():
    print('Testing started.')
    # Test case 1: Questions are the same.
    print('Testing case [1/3] started.')
    q1 = 'How to learn Python?'
    q2 = 'How to learn Python?'
    assert are_questions_similar(q1, q2) > 0.9, f'Test case [1/3] failed: Expected similarity for identical questions to be high.'

    # Test case 2: Questions are similar.
    print('Testing case [2/3] started.')
    q1 = 'What is the weather today?'
    q2 = 'Can you tell me today's weather?'
    assert are_questions_similar(q1, q2) > 0.7, f'Test case [2/3] failed: Expected similarity for similar questions to be high.'

    # Test case 3: Questions are different.
    print('Testing case [3/3] started.')
    q1 = 'When is the sunset?'
    q2 = 'How tall is the Eiffel Tower?'
    assert are_questions_similar(q1, q2) < 0.4, f'Test case [3/3] failed: Expected similarity for unrelated questions to be low.'
    print('Testing finished.')

# call_test_function_line --------------------

test_are_questions_similar()