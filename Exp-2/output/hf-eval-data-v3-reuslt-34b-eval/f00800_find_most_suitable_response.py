# function_import --------------------

from sentence_transformers import SentenceTransformer, util

# function_code --------------------

def find_most_suitable_response(query: str, docs: list) -> str:
    """
    Find the most suitable response to a user question from a list of responses provided.

    Args:
        query (str): The user's question.
        docs (list): A list of potential responses.

    Returns:
        str: The most suitable response.
    """
    
    # encode the input question and possible responses
    sent_embeddings = model.encode(docs, convert_to_tensor=True)
    
    # compute similarity scores for each possible response
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, sent_embeddings)[0]
    
    # find the index of the most similar response
    most_similar_response_index = cosine_scores.argmax()
    
    # return the corresponding element from the docs list
    return docs[most_similar_response_index]


# test_function_code --------------------

def test_find_most_suitable_response():
    assert find_most_suitable_response('How many people live in London?', ['Around 9 Million people live in London', 'London is known for its financial district']) == 'Around 9 Million people live in London'
    assert find_most_suitable_response('What is the capital of France?', ['Paris is the capital of France', 'France is known for its wine']) == 'Paris is the capital of France'
    assert find_most_suitable_response('Who won the world cup in 2018?', ['France won the world cup in 2018', 'The world cup is a football tournament']) == 'France won the world cup in 2018'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_find_most_suitable_response()