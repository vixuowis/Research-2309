# requirements_file --------------------

!pip install -U sentence-transformers

# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def compute_sentence_similarity(sentences):
    """Computes sentence embeddings for given sentences and returns similarity scores.

    Args:
        sentences (list): A list of sentences for which to compute embeddings.

    Returns:
        list: A list of floating-point numbers representing similarity scores.

    Raises:
        ValueError: If `sentences` is not a list of strings.
    """
    if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
        raise ValueError('Input must be a list of strings.')
    model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v2')
    embeddings = model.encode(sentences)
    return embeddings

# test_function_code --------------------

def test_compute_sentence_similarity():
    print("Testing started.")
    # Sample sentences for testing
    test_sentences = [
        "This is an example sentence for testing.",
        "Sentence similarity computation is very useful."
    ]
    
    # Testing case 1: Check if function returns a list
    print("Testing case [1/2] started.")
    results = compute_sentence_similarity(test_sentences)
    assert isinstance(results, list), f"Test case [1/2] failed: Function should return a list."

    # Testing case 2: Check if function raises ValueError for invalid input
    print("Testing case [2/2] started.")
    try:
        _ = compute_sentence_similarity('This is not a list')
        assert False, "Test case [2/2] failed: Function should raise ValueError for non-list input."
    except ValueError as e:
        assert str(e) == 'Input must be a list of strings.', f"Test case [2/2] failed: {e}"
    print("Testing finished.")
    return True

# call_test_function_line --------------------

test_compute_sentence_similarity()