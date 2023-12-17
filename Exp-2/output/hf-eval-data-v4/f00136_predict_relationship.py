# requirements_file --------------------

!pip install -U sentence_transformers transformers

# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def predict_relationship(sentence1, sentence2):
    """
    Determine the relationship between two sentences using a zero-shot classification model.

    Args:
    sentence1 (str): The first sentence.
    sentence2 (str): The second sentence.

    Returns:
    str: The predicted relationship ('contradiction', 'entailment', or 'neutral').
    """
    model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
    scores = model.predict([(sentence1, sentence2)])
    relationship = ['contradiction', 'entailment', 'neutral'][scores.argmax()]
    return relationship

# test_function_code --------------------

def test_predict_relationship():
    print("Testing predict_relationship function.")
    # Test case 1: Entailment
    print("Testing case [1/3] started.")
    assert predict_relationship('A man is eating pizza', 'A man eats something') == 'entailment', "Test case [1/3] failed: Expected 'entailment'."

    # Test case 2: Contradiction
    print("Testing case [2/3] started.")
    assert predict_relationship('A black race car starts up', 'A man is swimming') == 'contradiction', "Test case [2/3] failed: Expected 'contradiction'."

    # Test case 3: Neutral
    print("Testing case [3/3] started.")
    assert predict_relationship('A woman walks her dog', 'A woman goes for a run') == 'neutral', "Test case [3/3] failed: Expected 'neutral'."
    print("Testing finished.")

test_predict_relationship()