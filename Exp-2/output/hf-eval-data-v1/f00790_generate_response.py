from transformers import pipeline


def generate_response(message: str) -> str:
    """
    Generate a response to a given message using a conversational model.

    Args:
        message (str): The message to which the model should respond.

    Returns:
        str: The generated response.

    Raises:
        Exception: If there is an error in generating the response.
    """
    try:
        chatbot = pipeline('conversational', model='mywateriswet/ShuanBot')
        response = chatbot(message)
        return response
    except Exception as e:
        print(f'Error in generating response: {e}')
        raise