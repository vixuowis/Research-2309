from sentence_transformers import SentenceTransformer


def embed_movie_dialogues(movie_dialogues):
    """
    This function takes a list of movie dialogues as input and returns their dense vector representations.
    It uses the SentenceTransformer model from the Hugging Face Transformers library.
    
    Args:
    movie_dialogues (list): A list of movie dialogues.
    
    Returns:
    list: A list of dense vector representations of the movie dialogues.
    """
    # Initialize the SentenceTransformer model with the pre-trained model
    model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
    
    # Create dense vector representations of the movie dialogues
    embeddings = model.encode(movie_dialogues)
    
    return embeddings