# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def detect_conflicting_information(sentence_pairs):
    """
    Detects if the given sentence pairs contain conflicting information.

    Args:
        sentence_pairs (list): A list of tuples, where each tuple contains two sentences to be compared.

    Returns:
        list: A list of scores for each sentence pair. Each score is a list of three elements representing the probabilities of contradiction, entailment, and neutral respectively.
    """
    model = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')
    scores = model.predict(sentence_pairs)
    return scores

# test_function_code --------------------

def test_detect_conflicting_information():
    sentence_pairs = [
        ('A man is eating pizza', 'A man eats something'),
        ('A black race car starts up in front of a crowd of people.', 'A man is driving down a lonely road.'),
        ('A woman is playing violin', 'A woman is playing guitar')
    ]
    scores = detect_conflicting_information(sentence_pairs)
    assert len(scores) == len(sentence_pairs), 'The number of scores should be equal to the number of sentence pairs'
    for score in scores:
        assert len(score) == 3, 'Each score should have three elements'
        assert sum(score) == 1, 'The sum of the probabilities in each score should be 1'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_conflicting_information()