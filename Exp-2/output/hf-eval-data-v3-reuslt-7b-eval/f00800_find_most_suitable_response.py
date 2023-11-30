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
    # Load the BERT model for embedding sentences
    sentence_model = SentenceTransformer(model='bert-base-nli-mean-tokens')

    # Embed all docs into a single vector
    doc_embeddings = sentence_model.encode(docs, convert_to_tensor=True)
    
    # Embed the user question
    question_embedding = util.torch_utils.paired_cos_dist(question_sentence_embedding,doc_embeddings)[0]

    return docs[np.argmin(question_embedding)]

# test_function_code --------------------

def test_find_most_suitable_response():
    assert find_most_suitable_response('How many people live in London?', ['Around 9 Million people live in London', 'London is known for its financial district']) == 'Around 9 Million people live in London'
    assert find_most_suitable_response('What is the capital of France?', ['Paris is the capital of France', 'France is known for its wine']) == 'Paris is the capital of France'
    assert find_most_suitable_response('Who won the world cup in 2018?', ['France won the world cup in 2018', 'The world cup is a football tournament']) == 'France won the world cup in 2018'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_find_most_suitable_response()