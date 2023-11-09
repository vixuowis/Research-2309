from sentence_transformers import SentenceTransformer, util


def find_most_relevant_sentence(question: str, sentences: list) -> str:
    """
    This function finds the most relevant sentence among a list of sentences that answers a specific question.
    It uses the SentenceTransformer model from Hugging Face Transformers to compute the cosine similarity scores between the question and each of the sentences.
    The sentence with the highest similarity score is considered the best answer to the question.
    
    Args:
    question (str): The question to be answered.
    sentences (list): The list of sentences to find the answer from.
    
    Returns:
    str: The most relevant sentence that answers the question.
    """
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    question_emb = model.encode(question)
    sentences_emb = model.encode(sentences)
    scores = util.dot_score(question_emb, sentences_emb)
    best_sentence_index = scores.argmax()
    return sentences[best_sentence_index]