# requirements_file --------------------

!pip install -U sentence-transformers

# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def generate_movie_dialogue_embeddings(dialogues):
    """
    Generate dense vector embeddings for movie dialogue texts using a pre-trained BERT model.

    Args:
        dialogues (List[str]): A list of movie dialogue strings to be embedded.

    Returns:
        List[np.ndarray]: A list of numpy arrays, each representing the vector embedding of the corresponding dialogue.

    Raises:
        ValueError: If the 'dialogues' input is not a list of strings.
    """
    if not isinstance(dialogues, list) or not all(isinstance(d, str) for d in dialogues):
        raise ValueError("'dialogues' must be a list of strings.")
    model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
    embeddings = model.encode(dialogues)
    return embeddings

# test_function_code --------------------

def test_generate_movie_dialogue_embeddings():
    print("Testing started.")
    dialogues = ["I'll be back.", "May the Force be with you.", "You talking to me?"]
    embeddings = generate_movie_dialogue_embeddings(dialogues)

    # Testing case 1: Check if embeddings are generated
    print("Testing case [1/1] started.")
    assert isinstance(embeddings, list) and all(isinstance(e, np.ndarray) for e in embeddings), "Test case [1/1] failed: The function did not return a list of numpy arrays."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_movie_dialogue_embeddings()