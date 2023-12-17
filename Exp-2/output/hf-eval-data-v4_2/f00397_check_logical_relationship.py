# requirements_file --------------------

!pip install -U sentence_transformers transformers

# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def check_logical_relationship(sentence1, sentence2):
    """Check the logical relationship between two input sentences.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.
    Returns:
        dict: A dictionary with keys 'contradiction', 'entailment', 'neutral',
              and their corresponding probability scores.
    Raises:
        ValueError: If any of the inputs is not a string.
    """
    if not isinstance(sentence1, str) or not isinstance(sentence2, str):
        raise ValueError('Both inputs must be strings.')
    model = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')
    scores = model.predict([(sentence1, sentence2)])
    return {'contradiction': scores[0][0], 'entailment': scores[0][1], 'neutral': scores[0][2]}

# test_function_code --------------------

def test_check_logical_relationship():
    print("Testing started.")
    # Define test cases
    test_cases = [
        ('A man is eating pizza', 'A man eats something'),
        ('A black race car starts up in front of a crowd of people.', 'A man is driving down a lonely road.'),
        ('A woman is walking her dog.', 'A woman is at a park.')
    ]

    for i, (sentence1, sentence2) in enumerate(test_cases, start=1):
        print(f"Testing case [{i}/{len(test_cases)}] started.")
        result = check_logical_relationship(sentence1, sentence2)
        assert isinstance(result, dict) and 'contradiction' in result, f"Test case [{i}/{len(test_cases)}] failed: The result must be a dictionary with a key 'contradiction'."
        assert 'entailment' in result and 'neutral' in result, f"Test case [{i}/{len(test_cases)}] failed: The result must contain 'entailment' and 'neutral' keys."
    print("Testing finished.")

# call_test_function_line --------------------

test_check_logical_relationship()