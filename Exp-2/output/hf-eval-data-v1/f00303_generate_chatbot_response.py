from transformers import pipeline


def generate_chatbot_response(input_prompt):
    """
    This function uses the PygmalionAI/pygmalion-1.3b model from Hugging Face Transformers to generate a response to a given input prompt.
    The input prompt should include the character persona, dialogue history, and the user input message.
    
    Args:
        input_prompt (str): The input prompt for the chatbot.
    
    Returns:
        str: The generated response from the chatbot.
    """
    # Create a text-generation model using the pipeline function
    chatbot = pipeline('text-generation', 'PygmalionAI/pygmalion-1.3b')
    
    # Generate a response from the chatbot
    response = chatbot(input_prompt)
    
    return response