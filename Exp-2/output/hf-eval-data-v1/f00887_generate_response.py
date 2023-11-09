from transformers import pipeline

def generate_response(question: str) -> str:
    """
    Generate a human-like response to a given question using a pre-trained text generation model.

    Args:
        question (str): The customer's question.

    Returns:
        str: The generated response.

    Raises:
        ValueError: If the input question is not a string.
    """
    if not isinstance(question, str):
        raise ValueError('Input question must be a string.')

    # Load the pre-trained text generation model
    generator = pipeline('text-generation', model='facebook/opt-350m')

    # Generate a response to the question
    response = generator(question)

    return response[0]['generated_text']