# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def predict_relationship(sentence1: str, sentence2: str) -> str:
    """
    Predicts the relationship between two sentences using a pre-trained model.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        str: The predicted relationship between the two sentences. It can be 'contradiction', 'entailment', or 'neutral'.
    """
    model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
    scores = model.predict([(sentence1, sentence2)])
    relationship = ['contradiction', 'entailment', 'neutral'][scores.argmax()]
    return relationship

# test_function_code --------------------

def test_predict_relationship():
    """
    Tests the predict_relationship function.
    """
    sentence1 = 'A man is eating pizza'
    sentence2 = 'A man eats something'
    assert predict_relationship(sentence1, sentence2) in ['contradiction', 'entailment', 'neutral']
    sentence1 = 'A black race car starts up in front of a crowd of people.'
    sentence2 = 'A man is driving down a lonely road.'
    assert predict_relationship(sentence1, sentence2) in ['contradiction', 'entailment', 'neutral']

# call_test_function_code --------------------

test_predict_relationship()