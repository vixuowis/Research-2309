# requirements_file --------------------

pip install -U sentence-transformers

# function_import --------------------

from sentence_transformers import SentenceTransformer, util

# function_code --------------------

def find_best_response(query, responses):
    """
    Find the most suitable response to a user question from a list of provided responses.

    Args:
        query (str): The user question to find an answer for.
        responses (List[str]): The list of responses to choose from.

    Returns:
        Tuple[str, float]: The best matching response and the corresponding similarity score.

    Raises:
        ValueError: If the responses list is empty.
    """
    if not responses:
        raise ValueError('The list of responses cannot be empty.')

    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    query_emb = model.encode(query)
    response_embs = model.encode(responses)
    scores = util.dot_score(query_emb, response_embs)[0].cpu().tolist()
    best_response, best_score = max(zip(responses, scores), key=lambda x: x[1])
    return best_response, best_score

# test_function_code --------------------

def test_find_best_response():
    print('Testing started.')
    test_query = 'How many people live in London?'
    test_responses = ['About 9 Million people live in London.',
                      'London is known for its financial district.',
                      'London has a rich history dating back to Roman times.']

    # Test case 1
    print('Testing case [1/1] started.')
    best_response, best_score = find_best_response(test_query, test_responses)
    assert best_response == 'About 9 Million people live in London.', 'Test case [1/1] failed: The best response is incorrect.'
    print('Best response:', best_response)
    print('Similarity score:', best_score)
    print('Testing finished.')

# call_test_function_line --------------------

test_find_best_response()