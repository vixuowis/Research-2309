# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_customer_feedback(feedback_text):
    """
    Analyze the sentiment of the given customer feedback text in Spanish.

    Parameters:
    feedback_text (str): The text of the customer feedback in Spanish.

    Returns:
    dict: The analysis results, including the sentiment label and score.
    """
    # Define the model path for the sentiment analysis
    model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    # Load the sentiment analysis pipeline with the specified model
    sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

    # Perform sentiment analysis
    results = sentiment_task(feedback_text)

    # Return the analysis results
    return results[0]

# test_function_code --------------------

def test_analyze_customer_feedback():
    print("Testing analyze_customer_feedback function.")

    # Example Spanish customer feedback
    positive_feedback = "Me encanta este producto!"
    negative_feedback = "No estoy satisfecho con el servicio."
    neutral_feedback = "Es un producto normal, nada especial."

    # Test positive feedback
    print("Testing positive feedback case.")
    positive_result = analyze_customer_feedback(positive_feedback)
    assert positive_result['label'] == 'LABEL_2', f"Test failed for positive feedback: {positive_result}"

    # Test negative feedback
    print("Testing negative feedback case.")
    negative_result = analyze_customer_feedback(negative_feedback)
    assert negative_result['label'] == 'LABEL_0', f"Test failed for negative feedback: {negative_result}"

    # Test neutral feedback
    print("Testing neutral feedback case.")
    neutral_result = analyze_customer_feedback(neutral_feedback)
    assert neutral_result['label'] == 'LABEL_1', f"Test failed for neutral feedback: {neutral_result}"

    print("All tests passed.")

# Run the test function
test_analyze_customer_feedback()