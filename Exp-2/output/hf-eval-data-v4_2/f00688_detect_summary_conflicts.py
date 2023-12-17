# requirements_file --------------------

!pip install -U sentence_transformers transformers

# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def detect_summary_conflicts(summary_sentences):
    """
    Detects conflicting information in a summary of a book.

    Args:
        summary_sentences (list of tuples): A list of tuples, where each tuple contains a pair of sentences from the summary to be compared.

    Returns:
        list: A list of tuples containing the conflicting sentence pairs and their corresponding contradiction scores.

    Raises:
        ValueError: If summary_sentences is not a list of tuples.
    """
    # Validate input format
    if not all(isinstance(pair, tuple) for pair in summary_sentences):
        raise ValueError('Input must be a list of tuples.')

    # Instantiate the CrossEncoder model
    model = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')

    # Make predictions on the sentence pairs
    predictions = model.predict(summary_sentences)

    # Filter out sentence pairs with high contradiction scores
    conflicting_info = [(pair, score) for pair, score in zip(summary_sentences, predictions) if score[0] > 0.5]

    return conflicting_info

# test_function_code --------------------

def test_detect_summary_conflicts():
    print("Testing started.")

    # Test case 1: Valid input format
    print("Testing case [1/3] started.")
    summary = [('The protagonist is a wizard.', 'The protagonist is a magician.'),
               ('The setting is during the medieval times.', 'The story takes place in modern-day New York.'),
               ('The antagonist is defeated at the end.', 'The antagonist wins in the end.')]
    results = detect_summary_conflicts(summary)
    assert len(results) > 0, f"Test case [1/3] failed: Expected conflicting information, got {results}"

    # Test case 2: Invalid input format
    print("Testing case [2/3] started.")
    invalid_summary = ['This is not a tuple.']
    try:
        detect_summary_conflicts(invalid_summary)
        assert False, 'Test case [2/3] failed: ValueError not raised for invalid input format'
    except ValueError:
        pass

    # Test case 3: No conflicting information
    print("Testing case [3/3] started.")
    non_conflicting_summary = [('The hero saves the day.', 'The hero prevents a disaster.')]
    non_conflicting_results = detect_summary_conflicts(non_conflicting_summary)
    assert len(non_conflicting_results) == 0, f"Test case [3/3] failed: No conflicts expected, got {non_conflicting_results}"
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_summary_conflicts()