# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def embed_movie_dialogues(movie_dialogues):
    """
    This function takes a list of movie dialogues as input and returns their dense vector representations.

    Args:
        movie_dialogues (list): A list of movie dialogues.

    Returns:
        embeddings (list): A list of dense vector representations of the movie dialogues.
    """
    model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
    embeddings = model.encode(movie_dialogues)
    return embeddings

# test_function_code --------------------

def test_embed_movie_dialogues():
    """
    This function tests the embed_movie_dialogues function.
    """
    movie_dialogues = ["Dialogue from movie 1", "Dialogue from movie 2"]
    embeddings = embed_movie_dialogues(movie_dialogues)
    assert len(embeddings) == len(movie_dialogues), 'Number of embeddings does not match number of dialogues'
    assert len(embeddings[0]) == 768, 'Embedding dimension does not match expected dimension'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_embed_movie_dialogues()