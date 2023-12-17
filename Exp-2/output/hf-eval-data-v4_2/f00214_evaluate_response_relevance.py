# requirements_file --------------------

!pip install -U sentence_transformers transformers

# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def evaluate_response_relevance(customer_question: str, assistant_answer: str) -> str:
    """
    Evaluate the relevance of the assistant's response to the customer's question using a pre-trained NLI model.

    Args:
        customer_question: The question asked by the customer.
        assistant_answer: The answer given by the assistant.

    Returns:
        A string indicating the relation as 'contradictory', 'neutral', or 'entails'.

    Raises:
        ValueError: If the input questions or answers are not strings or are empty.
    """
    # Validate input arguments
    if not isinstance(customer_question, str) or not customer_question:
        raise ValueError('Invalid customer question. It must be a non-empty string.')
    if not isinstance(assistant_answer, str) or not assistant_answer:
        raise ValueError('Invalid assistant answer. It must be a non-empty string.')

    # Initialize the CrossEncoder model
    model = CrossEncoder('cross-encoder/nli-deberta-v3-small')

    # Predict the relation between the customer question and assistant answer
    scores = model.predict([(customer_question, assistant_answer)])

    # Relation categories
    categories = ['contradictory', 'neutral', 'entails']

    # Determine the highest score and corresponding relation
    max_score_index = scores[0].argmax()
    relation = categories[max_score_index]

    return relation

# test_function_code --------------------

def test_evaluate_response_relevance():
    print("Testing started.")
    # Test cases with pre-defined questions and answers
    test_cases = [
        ("What is the refund policy?", "We offer a 30-day money-back guarantee on all purchases.", "entails"),
        ("Do you provide technical support?", "Our helpline is available 24/7.", "entails"),
        ("How to cancel my subscription?", "Refer to our website for the latest news.", "contradictory")
    ]

    # Iterate through test cases
    for i, (question, answer, expected_relation) in enumerate(test_cases, start=1):
        print(f"Testing case [{i}/{len(test_cases)}] started.")
        actual_relation = evaluate_response_relevance(question, answer)
        assert actual_relation == expected_relation, f"Test case [{i}/{len(test_cases)}] failed: Expected '{{expected_relation}}', got '{{actual_relation}}'."

    print("Testing finished.")

# Call the test function
test_evaluate_response_relevance()

# call_test_function_line --------------------

test_evaluate_response_relevance()