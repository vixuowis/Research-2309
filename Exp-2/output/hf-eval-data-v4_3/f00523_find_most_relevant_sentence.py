# requirements_file --------------------

import subprocess

requirements = ["sentence-transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from sentence_transformers import SentenceTransformer, util

# function_code --------------------

def find_most_relevant_sentence(question, sentences):
    """
    Finds the most relevant sentence to a given question from a list of sentences using semantic similarity.

    Args:
        question (str): A question string.
        sentences (list): A list of sentence strings.

    Returns:
        str: The most relevant sentence from the list.

    Raises:
        ValueError: If the input question is not a string or sentences is not a list of strings.
    """
    if not isinstance(question, str) or not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
        raise ValueError('Invalid input types for question and sentences.')

    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    question_emb = model.encode(question)
    sentences_emb = model.encode(sentences)
    scores = util.dot_score(question_emb, sentences_emb)
    best_sentence_index = scores.argmax()
    return sentences[best_sentence_index]

# test_function_code --------------------

def test_find_most_relevant_sentence():
    print("Testing started.")
    question = "What is the main purpose of photosynthesis?"
    sentences = [
        "Photosynthesis is the process used by plants to convert light energy into chemical energy to fuel their growth.",
        "The Eiffel Tower is a famous landmark in Paris.",
        "Photosynthesis also produces oxygen as a byproduct, which is necessary for life on Earth."
    ]
    expected_answer = sentences[0]

    print("Testing case [1/1] started.")
    result = find_most_relevant_sentence(question, sentences)
    assert result == expected_answer, f"Test case [1/1] failed: Expected {{expected_answer}}, got {{result}}."
    print("Testing finished.")

# call_test_function_line --------------------

test_find_most_relevant_sentence()