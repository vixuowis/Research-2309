# requirements_file --------------------

!pip install -U transformers==4.3.0

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_customer_support_response(question_text):
    # Initialize the text generation pipeline with the specified model
    generator = pipeline('text-generation', model='facebook/opt-350m')

    # Generate a response to the customer's question
    responses = generator(question_text, num_return_sequences=1)

    # Return the first generated response
    return responses[0]['generated_text']

# test_function_code --------------------

def test_generate_customer_support_response():
    print("Testing started.")

    # Test case 1: Standard question
    question_1 = 'What is your return policy?'
    print("Testing case [1/3] started.")
    response_1 = generate_customer_support_response(question_1)
    assert response_1, f"Test case [1/3] failed: No response generated for question: {question_1}"

    # Test case 2: Technical support question
    question_2 = 'How do I reset my password?'
    print("Testing case [2/3] started.")
    response_2 = generate_customer_support_response(question_2)
    assert response_2, f"Test case [2/3] failed: No response generated for question: {question_2}"

    # Test case 3: Product inquiry
    question_3 = 'Do you have any vegan options?'
    print("Testing case [3/3] started.")
    response_3 = generate_customer_support_response(question_3)
    assert response_3, f"Test case [3/3] failed: No response generated for question: {question_3}"
    print("Testing finished.")

test_generate_customer_support_response()