# requirements_file --------------------

!pip install -U sentence-transformers

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def find_most_relevant_answer(question, answers):
    """
    Find the most relevant answer to the question by encoding text and computing cosine similarity.

    Args:
        question (str): The question text.
        answers (List[str]): A list of potential answers.

    Returns:
        str: The most relevant answer based on cosine similarity.

    Raises:
        ValueError: If `answers` list is empty.
    """
    if not answers:
        raise ValueError("The answer list is empty.")

    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
    question_embedding = model.encode(question)
    answer_embeddings = model.encode(answers)
    cos_sim_scores = cosine_similarity([question_embedding], answer_embeddings)[0]
    best_answer_index = cos_sim_scores.argmax()
    return answers[best_answer_index]

# test_function_code --------------------

def test_find_most_relevant_answer():
    print("Testing started.")
    # Prepare test data
    question = "What technology is used for natural language processing?"
    answers = [
        "Natural language processing involves machine learning.",
        "It's a sunny day outside.",
        "Basketball is a sport played by two teams.",
        "Language models are part of NLP.",
        "Apples are tasty fruits."
    ]

    # Testing case 1: Find the most relevant answer
    print("Testing case [1/1] started.")
    expected_answer = "Natural language processing involves machine learning."
    result = find_most_relevant_answer(question, answers)
    assert result == expected_answer, f"Test case [1/1] failed: Expected {{expected_answer}}, got {{result}}"
    print("Testing finished.")

# call_test_function_line --------------------

test_find_most_relevant_answer()