# function_import --------------------

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def suggest_questions(user_question, available_questions):
    """
    This function suggests questions similar to the one submitted by the user.
    
    Args:
    user_question (str): The question submitted by the user.
    available_questions (list): The list of questions available in the database.
    
    Returns:
    list: A list of suggested questions.
    """
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    user_question_embedding = model.encode([user_question])
    available_questions_embeddings = model.encode(available_questions)
    similarities = cosine_similarity(user_question_embedding, available_questions_embeddings)
    top_5_similar_questions_indices = np.argsort(similarities[0])[-5:]
    return [available_questions[i] for i in top_5_similar_questions_indices]

# test_function_code --------------------

def test_suggest_questions():
    """
    This function tests the suggest_questions function.
    """
    user_question = 'What is your favorite color?'
    available_questions = ['What is your favorite movie?', 'What is your favorite food?', 'What is your favorite book?', 'What is your favorite sport?', 'What is your favorite song?', 'What is your favorite animal?', 'What is your favorite place?', 'What is your favorite car?', 'What is your favorite drink?', 'What is your favorite game?']
    suggested_questions = suggest_questions(user_question, available_questions)
    assert len(suggested_questions) == 5
    assert all([question in available_questions for question in suggested_questions])

# call_test_function_code --------------------

test_suggest_questions()