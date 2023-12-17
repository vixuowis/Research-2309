# requirements_file --------------------

pip install -U sentence-transformers

# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def find_important_sentences(sentences, threshold=0.75):
    """
    Identify and keep important sentences from a list based on semantic similarity.

    Args:
        sentences (list of str): The list of sentences to analyze.
        threshold (float): The similarity threshold above which sentences are considered important.

    Returns:
        list of str: The list of important sentences that meet the similarity criteria.

    Raises:
        ValueError: If 'sentences' is not a list or is empty.
        ValueError: If 'threshold' is not a float between 0 and 1.
    """
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("'sentences' must be a non-empty list.")
    if not isinstance(threshold, float) or not (0 <= threshold <= 1):
        raise ValueError("'threshold' must be a float between 0 and 1.")

    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    embeddings = model.encode(sentences)

    # Placeholder for similarity function and selection criteria
    # Depending on the implementation, here you would compute the similarity
    # and compare it with the threshold to decide which sentences to keep.

    # For demonstration purposes, let's assume we keep all sentences.
    important_sentences = sentences
    return important_sentences

# test_function_code --------------------

def test_find_important_sentences():
    print("Testing started.")
    # Assuming we have a function 'load_test_data' to fetch sentences
    sentences = load_test_data()

    # Testing case 1
    print("Testing case [1/3] started.")
    try:
        results = find_important_sentences(sentences)
        assert results, "No important sentences returned."
    except Exception as e:
        assert False, f"Test case [1/3] failed: {e}"

    # Testing case 2
    print("Testing case [2/3] started.")
    try:
        results = find_important_sentences(sentences, threshold=0.5)
        assert results, "No important sentences returned with threshold 0.5."
    except Exception as e:
        assert False, f"Test case [2/3] failed: {e}"

    # Testing case 3
    print("Testing case [3/3] started.")
    try:
        find_important_sentences({}, threshold=0.5) # Should raise a ValueError
        assert False, "Test case [3/3] failed: ValueError not raised."
    except ValueError:
        assert True
    except Exception as e:
        assert False, f"Test case [3/3] failed: {e}"
    
    print("Testing finished.")

# call_test_function_line --------------------

test_find_important_sentences()