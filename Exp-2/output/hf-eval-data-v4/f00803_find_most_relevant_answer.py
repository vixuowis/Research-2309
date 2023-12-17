# requirements_file --------------------

!pip install -U sentence-transformers sklearn

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def find_most_relevant_answer(question, answers):
    """
    Find the most relevant answer to a specific question from a list of answers
    using sentence similarity.
    
    Parameters:
        question (str): The question to be answered.
        answers (list): A list of potential answers.
    
    Returns:
        str: The most relevant answer.
    """
    # Initialize Sentence Transformer model
    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
    
    # Encode the question and answers into embeddings
    question_embedding = model.encode(question)
    answer_embeddings = model.encode(answers)

    # Compute the cosine similarity scores
    cos_sim_scores = cosine_similarity([question_embedding], answer_embeddings)

    # Find the index of the best answer (highest cosine similarity score)
    best_answer_index = cos_sim_scores.argmax()
    best_answer = answers[best_answer_index]

    return best_answer

# test_function_code --------------------

def test_find_most_relevant_answer():
    print("Testing started.")
    # Test case: a simple question and three potential answers
    question = "What is the capital of France?"
    answers = ["Berlin", "Paris", "Lisbon"]

    print("Testing case [1/1] started.")
    most_relevant_answer = find_most_relevant_answer(question, answers)
    assert most_relevant_answer == "Paris", f"Test case failed: Expected 'Paris' but got {most_relevant_answer}"
    print("Testing finished.")

# Run the test function
test_find_most_relevant_answer()