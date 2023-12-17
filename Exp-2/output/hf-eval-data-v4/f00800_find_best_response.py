# requirements_file --------------------

!pip install -U sentence-transformers

# function_import --------------------

from sentence_transformers import SentenceTransformer, util

# function_code --------------------

def find_best_response(query, responses):
    """
    Find the most suitable response to a user question from a list of responses.

    :param query: A string representing the user's question.
    :param responses: A list of strings representing potential responses.
    :return: A tuple of the best response and its similarity score.
    """
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    query_emb = model.encode(query)
    response_embs = model.encode(responses)
    scores = util.dot_score(query_emb, response_embs)[0].cpu().tolist()
    best_response = max(zip(responses, scores), key=lambda rs: rs[1])
    return best_response

# test_function_code --------------------

def test_find_best_response():
    print("Testing find_best_response function.")
    query = 'How many people live in London?'
    responses = ['Around 9 Million people live in London.', 'London is known for its financial district.']
    expected_response, expected_score = ('Around 9 Million people live in London.', 0.9)  # Example of expected response with a high arbitrary similarity score
    best_response, score = find_best_response(query, responses)
    assert best_response == expected_response, f"Expected {expected_response}, got {best_response}"
    assert score > 0.8, f"Expected score > 0.8, got {score}"  # Assuming a threshold for the score
    print("All tests passed.")

test_find_best_response()