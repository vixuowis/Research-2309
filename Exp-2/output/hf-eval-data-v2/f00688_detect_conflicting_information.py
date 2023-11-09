# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def detect_conflicting_information(summary):
    """
    Detects if the given summary of a book contains conflicting information.
    
    Args:
        summary (list): A list of tuples. Each tuple contains a pair of sentences from the summary.
    
    Returns:
        dict: A dictionary where keys are the sentence pairs and values are the prediction scores. High contradiction scores for a pair of sentences indicate conflicting information.
    
    Raises:
        ValueError: If the summary is not a list or if it's empty.
    """
    if not isinstance(summary, list) or not summary:
        raise ValueError('Summary should be a non-empty list of sentence pairs.')
    
    model = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')
    scores = model.predict(summary)
    
    return dict(zip(summary, scores))

# test_function_code --------------------

def test_detect_conflicting_information():
    """
    Tests the function detect_conflicting_information.
    """
    summary = [
        ('A man is eating pizza', 'A man eats something'),
        ('A black race car starts up in front of a crowd of people.', 'A man is driving down a lonely road.')
    ]
    result = detect_conflicting_information(summary)
    
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert len(result) == len(summary), 'The result should have the same length as the input summary.'
    for pair, score in result.items():
        assert isinstance(score, float), 'Each score should be a float.'
        assert 0 <= score <= 1, 'Each score should be between 0 and 1.'

# call_test_function_code --------------------

test_detect_conflicting_information()