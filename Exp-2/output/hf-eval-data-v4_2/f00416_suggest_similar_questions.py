# requirements_file --------------------

pip install -U sentence-transformers

# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def suggest_similar_questions(user_question, available_questions):
    """
    Suggests questions similar to the user's submitted question.

    Args:
        user_question (str): The question submitted by the user.
        available_questions (list): A list of pre-existing questions to match against.

    Returns:
        list: A list of questions that are semantically similar to the user's question.

    Raises:
        ValueError: If user_question is not a string or available_questions is not a list of strings.
    """
    if not isinstance(user_question, str) or not all(isinstance(q, str) for q in available_questions):
        raise ValueError('Input must be a string (user_question) and a list of strings (available_questions)')

    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    user_question_embedding = model.encode([user_question])
    available_questions_embeddings = model.encode(available_questions)

    # Here you would implement the logic to find the most similar questions
    # For example purpose, we assume a function `find_most_similar` exists
    similar_questions = find_most_similar(user_question_embedding, available_questions_embeddings)

    return similar_questions

# test_function_code --------------------

def test_suggest_similar_questions():
    print("Testing started.")
    # Here you would load a dataset if needed, for testing purpose we use predefined questions
    user_question = 'What are your hobbies?'
    available_questions = ['What do you like to do for fun?', 'What are your interests?', 'Tell me about your hobbies.']

    # Testing case 1: Test with a valid user question and available questions
    print("Testing case [1/1] started.")
    suggested_questions = suggest_similar_questions(user_question, available_questions)
    assert len(suggested_questions) > 0, f"Test case [1/1] failed: No suggestions returned."
    print("Testing finished.")

# call_test_function_line --------------------

test_suggest_similar_questions()