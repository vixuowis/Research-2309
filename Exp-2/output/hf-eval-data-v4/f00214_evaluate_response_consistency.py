# requirements_file --------------------

!pip install -U sentence_transformers, transformers

# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def evaluate_response_consistency(customer_question, assistant_answer):
    """
    Evaluates the consistency of the response given by the assistant to the customer's question.

    Parameters:
    customer_question (str): The customer's question.
    assistant_answer (str): The assistant's answer.

    Returns:
    dict: A dictionary with scores for 'contradiction', 'entailment', and 'neutral'.
    """
    model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
    scores = model.predict([(customer_question, assistant_answer)])
    return {
        'contradiction': scores[0][0],
        'entailment': scores[0][1],
        'neutral': scores[0][2]
    }

# test_function_code --------------------

def test_evaluate_response_consistency():
    print("Testing evaluate_response_consistency function.")

    # Test case 1: Expect a high entailment score
    question = "What is the refund policy?"
    answer = "We offer a 30-day money-back guarantee on all purchases."
    scores = evaluate_response_consistency(question, answer)
    assert scores['entailment'] > scores['contradiction'] and scores['entailment'] > scores['neutral'], "Test case 1 failed: Expected a high entailment score."

    # Test case 2: Expect a high contradiction score
    question = "Can I pay with Bitcoin?"
    answer = "We only accept credit card and PayPal payments."
    scores = evaluate_response_consistency(question, answer)
    assert scores['contradiction'] > scores['entailment'] and scores['contradiction'] > scores['neutral'], "Test case 2 failed: Expected a high contradiction score."

    # Test case 3: Expect a high neutral score
    question = "How long does shipping take?"
    answer = "Orders are processed within 24 hours."
    scores = evaluate_response_consistency(question, answer)
    assert scores['neutral'] > scores['contradiction'] and scores['neutral'] > scores['entailment'], "Test case 3 failed: Expected a high neutral score."

    print("All test cases passed for evaluate_response_consistency function.")

# Run the test function
test_evaluate_response_consistency()