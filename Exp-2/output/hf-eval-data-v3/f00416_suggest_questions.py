# function_import --------------------

from sentence_transformers import SentenceTransformer
import numpy as np

# function_code --------------------

def suggest_questions(user_question, available_questions):
    """
    Suggests questions similar to the user's submitted question.

    Args:
        user_question (str): The question submitted by the user.
        available_questions (list): A list of available questions in the database.

    Returns:
        list: A list of suggested questions.
    """
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    user_question_embedding = model.encode([user_question])
    available_questions_embeddings = model.encode(available_questions)
    similarities = np.inner(user_question_embedding, available_questions_embeddings)
    top_5_indices = np.argsort(similarities, axis=-1, kind='quicksort', order=None)[::-1][:5]
    return [available_questions[i] for i in top_5_indices]

# test_function_code --------------------

def test_suggest_questions():
    user_question = 'What is your favorite color?'
    available_questions = ['What is your favorite food?', 'What is your favorite movie?', 'What is your favorite book?', 'What is your favorite sport?', 'What is your favorite song?', 'What is your favorite animal?', 'What is your favorite place?', 'What is your favorite game?', 'What is your favorite hobby?', 'What is your favorite season?']
    suggestions = suggest_questions(user_question, available_questions)
    assert isinstance(suggestions, list), 'The result is not a list.'
    assert len(suggestions) == 5, 'The number of suggestions is not 5.'
    for suggestion in suggestions:
        assert suggestion in available_questions, 'The suggested question is not in the available questions.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_suggest_questions()