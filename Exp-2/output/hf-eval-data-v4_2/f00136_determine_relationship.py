# requirements_file --------------------

!pip install -U sentence_transformers

# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def determine_relationship(sentence1, sentence2):
    """
    Determines the relationship between two sentences using zero-shot classification.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        str: The predicted relationship ('contradiction', 'entailment', 'neutral').

    Raises:
        ValueError: If any of the sentences is empty.
    """
    if not sentence1 or not sentence2:
        raise ValueError('Input sentences cannot be empty.')

    model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
    scores = model.predict([(sentence1, sentence2)])
    relationship = ['contradiction', 'entailment', 'neutral'][scores.argmax()]
    return relationship

# test_function_code --------------------

def test_determine_relationship():
    print("Testing started.")

    # Test case 1: Contradictory sentences
    print("Testing case [1/3] started.")
    assert determine_relationship('The sky is blue', 'The sky is green') == 'contradiction', f"Test case [1/3] failed: Contradiction not detected."

    # Test case 2: Entailing sentences
    print("Testing case [2/3] started.")
    assert determine_relationship('A man is eating food', 'Someone is eating') == 'entailment', f"Test case [2/3] failed: Entailment not detected."

    # Test case 3: Neutral sentences
    print("Testing case [3/3] started.")
    assert determine_relationship('A cat is sitting on the mat', 'A dog is playing in the yard') == 'neutral', f"Test case [3/3] failed: Neutrality not detected."
    print("Testing finished.")

# call_test_function_line --------------------

test_determine_relationship()