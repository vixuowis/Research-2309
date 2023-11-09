# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def embed_movie_dialogues(movie_dialogues):
    """
    This function takes a list of movie dialogues as input and returns their dense vector representations.
    
    Args:
        movie_dialogues (list): A list of movie dialogues. Each dialogue is a string.
    
    Returns:
        embeddings (list): A list of dense vector representations of the movie dialogues. Each representation is a 768-dimensional vector.
    
    Raises:
        ValueError: If the input is not a list or if any of the elements in the list is not a string.
    """
    if not isinstance(movie_dialogues, list) or not all(isinstance(dialogue, str) for dialogue in movie_dialogues):
        raise ValueError('Input should be a list of strings.')
    
    model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
    embeddings = model.encode(movie_dialogues)
    return embeddings

# test_function_code --------------------

def test_embed_movie_dialogues():
    """
    This function tests the embed_movie_dialogues function by using a sample of movie dialogues.
    """
    movie_dialogues = ['I am your father.', 'May the Force be with you.', 'I have a bad feeling about this.']
    embeddings = embed_movie_dialogues(movie_dialogues)
    assert isinstance(embeddings, list), 'The output should be a list.'
    assert len(embeddings) == len(movie_dialogues), 'The number of embeddings should be equal to the number of dialogues.'
    assert all(isinstance(embedding, list) and len(embedding) == 768 for embedding in embeddings), 'Each embedding should be a 768-dimensional vector.'

# call_test_function_code --------------------

test_embed_movie_dialogues()