# function_import --------------------

from sentence_transformers import SentenceTransformer, util

# function_code --------------------

def find_most_relevant_sentence(question: str, sentences: list) -> str:
    """
    Find the most relevant sentence among a list of sentences that answers a specific question.

    Args:
        question (str): The question to be answered.
        sentences (list): A list of sentences among which the answer lies.

    Returns:
        str: The most relevant sentence that answers the question.
    """
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    question_emb = model.encode(question)
    sentences_emb = model.encode(sentences)
    scores = util.dot_score(question_emb, sentences_emb)
    best_sentence_index = scores.argmax()
    return sentences[best_sentence_index]

# test_function_code --------------------

def test_find_most_relevant_sentence():
    assert find_most_relevant_sentence('What is the main purpose of photosynthesis?', ['Photosynthesis is the process used by plants to convert light energy into chemical energy to fuel their growth.', 'The Eiffel Tower is a famous landmark in Paris.', 'Photosynthesis also produces oxygen as a byproduct, which is necessary for life on Earth.']) == 'Photosynthesis is the process used by plants to convert light energy into chemical energy to fuel their growth.'
    assert find_most_relevant_sentence('Who won the world cup in 2018?', ['France won the world cup in 2018.', 'The Eiffel Tower is a famous landmark in Paris.', 'Germany won the world cup in 2014.']) == 'France won the world cup in 2018.'
    assert find_most_relevant_sentence('What is the capital of France?', ['Paris is the capital of France.', 'France won the world cup in 2018.', 'The Eiffel Tower is a famous landmark in Paris.']) == 'Paris is the capital of France.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_find_most_relevant_sentence()