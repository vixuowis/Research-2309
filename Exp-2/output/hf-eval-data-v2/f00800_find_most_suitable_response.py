# function_import --------------------

from sentence_transformers import SentenceTransformer, util

# function_code --------------------

def find_most_suitable_response(query: str, docs: list) -> str:
    '''
    Find the most suitable response to a user question from a list of responses provided.

    Args:
    query (str): The user's question.
    docs (list): A list of potential responses.

    Returns:
    str: The most suitable response.
    '''
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    doc_score_pairs = list(zip(docs, scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    return doc_score_pairs[0][0]

# test_function_code --------------------

def test_find_most_suitable_response():
    '''
    Test the function find_most_suitable_response.
    '''
    query = 'How many people live in London?'
    docs = ['Around 9 Million people live in London', 'London is known for its financial district']
    assert find_most_suitable_response(query, docs) == 'Around 9 Million people live in London'

# call_test_function_code --------------------

test_find_most_suitable_response()