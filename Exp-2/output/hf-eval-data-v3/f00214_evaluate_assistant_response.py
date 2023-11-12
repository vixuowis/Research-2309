# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def evaluate_assistant_response(customer_question: str, assistant_answer: str) -> dict:
    """
    Evaluate the response of an AI assistant to a customer's question.

    Args:
        customer_question (str): The question asked by the customer.
        assistant_answer (str): The answer provided by the AI assistant.

    Returns:
        dict: A dictionary with the scores for contradiction, entailment, and neutral.
    """
    model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
    scores = model.predict([(customer_question, assistant_answer)])
    return {'contradiction': scores[0], 'entailment': scores[1], 'neutral': scores[2]}

# test_function_code --------------------

def test_evaluate_assistant_response():
    assert evaluate_assistant_response('What is the refund policy?', 'We offer a 30-day money-back guarantee on all purchases.')['contradiction'] < 0.5
    assert evaluate_assistant_response('Is the sky blue?', 'Yes, the sky is blue.')['entailment'] > 0.5
    assert evaluate_assistant_response('Do you sell cars?', 'We sell a variety of products, including cars.')['neutral'] > 0.5
    return 'All Tests Passed'

# call_test_function_code --------------------

test_evaluate_assistant_response()