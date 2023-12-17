# requirements_file --------------------

!pip install -U sentence-transformers scipy

# function_import --------------------

from sentence_transformers import SentenceTransformer
import scipy.spatial

# function_code --------------------

def suggest_similar_questions(user_question, available_questions, top_k=5):
    """
    Given a user's question, suggest top_k similar questions from a list of available questions.

    Parameters:
    user_question (str): The question submitted by the user.
    available_questions (list): A list of questions to compare against.
    top_k (int): Number of top similar questions to retrieve.

    Returns:
    list: A list of top_k similar questions.
    """
    # Initialize the model
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    
    # Encode the user question and available questions
    user_question_embedding = model.encode([user_question])
    available_questions_embeddings = model.encode(available_questions)

    # Calculate distances between the user question and all available questions
    distances = scipy.spatial.distance.cdist(user_question_embedding, available_questions_embeddings, "cosine")[0]
    
    # Get the indices of the questions with the smallest distances
    top_k_indices = distances.argsort()[:top_k]
    
    # Retrieve the most similar questions
    similar_questions = [available_questions[index] for index in top_k_indices]
    return similar_questions

# test_function_code --------------------

def test_suggest_similar_questions():
    print("Testing suggest_similar_questions function...")
    available_questions = [
        'What are your hobbies?',
        'How do you spend your free time?',
        'What is your favorite book?',
        'Do you enjoy outdoor activities?'
    ]
    user_question = 'What do you like to do for fun?'

    # Get suggested questions
    suggested_questions = suggest_similar_questions(user_question, available_questions, top_k=2)

    # Check if the function returns 2 suggestions
    assert len(suggested_questions) == 2, "Failed: The function did not return the correct number of suggestions."
    
    # Check if the suggestions are part of the available questions
    for question in suggested_questions:
        assert question in available_questions, f"Failed: Suggested question '{question}' is not in available questions."
    
    print("All tests passed.")

# Run the test function
test_suggest_similar_questions()