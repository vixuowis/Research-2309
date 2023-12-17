# requirements_file --------------------

!pip install -U sentence_transformers

# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def generate_movie_script_embeddings(movie_dialogues):
    """
    Generate dense vector representations for movie script dialogues.

    Parameters:
    movie_dialogues (list of str): Dialogues from movies to be embedded.

    Returns:
    list: List of embeddings, each corresponding to a movie dialogue.
    """
    model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
    embeddings = model.encode(movie_dialogues)
    return embeddings


# test_function_code --------------------

def test_generate_movie_script_embeddings():
    print("Testing started.")
    sample_data = [
        "May the Force be with you.",
        "I'm going to make him an offer he can't refuse.",
        "I feel the need - the need for speed!"
    ]

    print("Testing case [1/1] started.")
    embeddings = generate_movie_script_embeddings(sample_data)
    assert len(embeddings) == len(sample_data), f"Test case failed: The number of embeddings ({len(embeddings)}) does not match the number of input dialogues ({len(sample_data)})."
    print("Testing finished.")

# Run the test function
test_generate_movie_script_embeddings()
