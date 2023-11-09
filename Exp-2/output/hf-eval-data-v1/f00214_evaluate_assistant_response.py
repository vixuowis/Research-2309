from sentence_transformers import CrossEncoder


def evaluate_assistant_response(customer_question: str, assistant_answer: str) -> dict:
    """
    This function evaluates the response of an AI assistant to a customer's question.
    It uses a pre-trained model from Hugging Face Transformers to determine if the response is contradictory, neutral or entails the customer's question.
    
    Args:
    customer_question (str): The question asked by the customer.
    assistant_answer (str): The answer provided by the AI assistant.
    
    Returns:
    dict: A dictionary with the scores for contradiction, entailment, and neutral.
    """
    # Initialize the CrossEncoder model
    model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
    
    # Predict the relation between the question and the answer
    scores = model.predict([(customer_question, assistant_answer)])
    
    # Return the scores
    return {'contradiction': scores[0][0], 'entailment': scores[0][1], 'neutral': scores[0][2]}