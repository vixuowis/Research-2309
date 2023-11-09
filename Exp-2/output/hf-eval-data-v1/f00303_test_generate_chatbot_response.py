def test_generate_chatbot_response():
    """
    This function tests the generate_chatbot_response function.
    It uses a sample input prompt and checks if the output is a string.
    """
    # Define a sample input prompt
    input_prompt = (
        "CompanyBot's Persona: I am a helpful chatbot designed to answer questions about our products and services.\n"
        "You: What products do you offer?\n"
    )
    
    # Call the function with the sample input
    response = generate_chatbot_response(input_prompt)
    
    # Check if the output is a string
    assert isinstance(response, str), 'The output should be a string.'

test_generate_chatbot_response()