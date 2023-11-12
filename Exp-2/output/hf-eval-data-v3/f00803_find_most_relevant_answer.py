# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def find_most_relevant_answer(question: str, answers: list) -> str:
    """
    Find the most relevant answer to a specific question using sentence similarity.

    Args:
        question (str): The question to which we want to find the most relevant answer.
        answers (list): A list of potential answers.

    Returns:
        str: The most relevant answer to the question.
    """
    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
    question_embedding = model.encode(question)
    answer_embeddings = model.encode(answers)
    cos_sim_scores = cosine_similarity([question_embedding], answer_embeddings)
    best_answer_index = cos_sim_scores.argmax()
    best_answer = answers[best_answer_index]
    return best_answer

# test_function_code --------------------

def test_find_most_relevant_answer():
    assert find_most_relevant_answer('What is the capital of France?', ['Paris', 'London', 'Berlin']) == 'Paris'
    assert find_most_relevant_answer('Who is the president of the United States?', ['Joe Biden', 'Donald Trump', 'Barack Obama']) == 'Joe Biden'
    assert find_most_relevant_answer('What is the largest planet in the solar system?', ['Earth', 'Mars', 'Jupiter']) == 'Jupiter'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_find_most_relevant_answer()