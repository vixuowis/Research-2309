from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def find_most_relevant_answer(question: str, answers: list) -> str:
    """
    This function finds the most relevant answer to a specific question using sentence similarity.
    
    Args:
        question (str): The question to which we want to find the most relevant answer.
        answers (list): A list of potential answers.
    
    Returns:
        str: The most relevant answer to the question.
    
    Raises:
        ValueError: If the answers list is empty.
    """
    if not answers:
        raise ValueError('The answers list cannot be empty.')
    
    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
    
    question_embedding = model.encode(question)
    answer_embeddings = model.encode(answers)
    
    cos_sim_scores = cosine_similarity([question_embedding], answer_embeddings)
    best_answer_index = cos_sim_scores.argmax()
    
    return answers[best_answer_index]