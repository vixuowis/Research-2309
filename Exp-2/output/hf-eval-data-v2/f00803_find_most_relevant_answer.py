# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def find_most_relevant_answer(question: str, answers: list) -> str:
    """
    This function finds the most relevant answer to a specific question using sentence similarity.

    Args:
        question (str): The question to which we need to find the most relevant answer.
        answers (list): A list of potential answers.

    Returns:
        str: The most relevant answer to the question.
    """
    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
    question_embedding = model.encode(question)
    answer_embeddings = model.encode(answers)
    cos_sim_scores = cosine_similarity([question_embedding], answer_embeddings)
    best_answer_index = cos_sim_scores.argmax()
    return answers[best_answer_index]

# test_function_code --------------------

def test_find_most_relevant_answer():
    """
    This function tests the find_most_relevant_answer function.
    """
    question = 'What is the capital of France?'
    answers = ['Paris', 'London', 'Berlin']
    assert find_most_relevant_answer(question, answers) == 'Paris'
    question = 'Who is the president of the United States?'
    answers = ['Joe Biden', 'Donald Trump', 'Barack Obama']
    assert find_most_relevant_answer(question, answers) == 'Joe Biden'

# call_test_function_code --------------------

test_find_most_relevant_answer()