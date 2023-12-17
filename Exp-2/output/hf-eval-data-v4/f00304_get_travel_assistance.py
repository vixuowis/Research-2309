# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_travel_assistance(user_message):
    """
    Use a pre-trained BlenderBot 9B model to generate responses for travel-related questions.
    
    Parameters:
    user_message (str): The user's message or question about travel.
    
    Returns:
    str: The generated response from the chatbot.
    """
    # Initiate the conversational pipeline with BlenderBot 9B model
    chatbot = pipeline('conversational', model='hyunwoongko/blenderbot-9B')
    
    # Get the response for the user's message
    response = chatbot(user_message)
    
    # Extract and return the generated text
    return response[0]['generated_text']

# test_function_code --------------------

def test_get_travel_assistance():
    print("Testing started.")
    # Sample user messages to test the function
    test_cases = [
        "I'm planning a vacation to Italy. Can you suggest some must-visit places?",
        "What are the top attractions in Paris?",
        "Is it safe to travel to Egypt these days?"
    ]

    # Test each case to ensure that a response is generated
    for i, message in enumerate(test_cases, 1):
        print(f"Testing case [{i}/{len(test_cases)}] started.")
        response = get_travel_assistance(message)
        assert response is not None and isinstance(response, str), f"Test case [{i}/{len(test_cases)}] failed: The function did not return a string or returned None."
        print(f"Response to '{message}': {response}")
        print(f"Testing case [{i}/{len(test_cases)}] finished.")

    print("Testing finished.")

# Run the test function
test_get_travel_assistance()