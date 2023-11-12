# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_sentence_similarity(sentence1: str, sentence2: str) -> float:
    '''
    Calculate the similarity between two sentences using SentenceTransformer.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        float: The similarity score between the two sentences. The score is between -1 and 1.
    '''
    sentences = [sentence1, sentence2]
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
    return similarity

# test_function_code --------------------

def test_calculate_sentence_similarity():
    assert abs(calculate_sentence_similarity('I love going to the park', 'My favorite activity is visiting the park') - 0.9) < 0.1
    assert abs(calculate_sentence_similarity('I love going to the park', 'I hate going to the park') - 0.7) < 0.1
    assert abs(calculate_sentence_similarity('I love going to the park', 'I love eating ice cream') - 0.5) < 0.1
    return 'All Tests Passed'

# call_test_function_code --------------------

test_calculate_sentence_similarity()