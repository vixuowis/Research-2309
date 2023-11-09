# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def evaluate_assistant_response(question: str, answer: str) -> dict:
    """
    Evaluate the response of an AI assistant to a customer's question.

    Args:
        question (str): The customer's question.
        answer (str): The AI assistant's response.

    Returns:
        dict: A dictionary with the scores for contradiction, entailment, and neutral.
    """
    model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
    scores = model.predict([(question, answer)])
    return {'contradiction': scores[0][0], 'entailment': scores[0][1], 'neutral': scores[0][2]}

# test_function_code --------------------

def test_evaluate_assistant_response():
    """
    Test the function evaluate_assistant_response.
    """
    question = 'What is the refund policy?'
    answer = 'We offer a 30-day money-back guarantee on all purchases.'
    result = evaluate_assistant_response(question, answer)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'contradiction' in result, 'The result should have a contradiction score.'
    assert 'entailment' in result, 'The result should have an entailment score.'
    assert 'neutral' in result, 'The result should have a neutral score.'

# call_test_function_code --------------------

test_evaluate_assistant_response()