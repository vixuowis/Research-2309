# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_conversational_response(question: str) -> str:
    """
    Generates a conversational response using a pre-trained model.

    Args:
        question: A string containing the customer's question.

    Returns:
        A string containing the model's response.

    Raises:
        ValueError: If the input question is not a string.
    """
    if not isinstance(question, str):
        raise ValueError('The input question must be a string.')

    # Initialize the conversational pipeline with the pre-trained model
    conv_pipeline = pipeline('conversational', model='ingen51/DialoGPT-medium-GPT4')

    # Generate response
    return conv_pipeline(question)

# test_function_code --------------------

def test_generate_conversational_response():
    print("Testing started.")
    sample_questions = [
        "What is the warranty period for this product?",
        "How do I install this device?",
        "My product is not working, what should I do?"
    ]

    # Test cases
    for idx, question in enumerate(sample_questions):
        print(f"Testing case [{idx+1}/{len(sample_questions)}] started.")
        response = generate_conversational_response(question)
        assert isinstance(response, str), f"Test case [{idx+1}/{len(sample_questions)}] failed: response is not a string."
        assert response, f"Test case [{idx+1}/{len(sample_questions)}] failed: response is empty."
    print("Testing finished.")

# Run the test function


# call_test_function_line --------------------

test_generate_conversational_response()