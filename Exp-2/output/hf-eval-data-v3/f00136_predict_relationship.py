# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def predict_relationship(sentence1: str, sentence2: str) -> str:
    """
    Predicts the relationship between two sentences using a pre-trained model from Hugging Face Transformers.

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
    assert predict_relationship('The dog is playing in the park', 'The dog is having fun outdoors') == 'neutral'
    assert predict_relationship('A man is eating pizza', 'A man eats something') == 'entailment'
    assert predict_relationship('A black race car starts up in front of a crowd of people.', 'A man is driving down a lonely road.') == 'contradiction'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_relationship()