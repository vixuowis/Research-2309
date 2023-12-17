# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_customer_feedback(feedback):
    """
    Analyzes sentiment of Spanish customer feedback using BETO sentiment analysis model.
    :param feedback: str - Customer feedback text in Spanish
    :return: dict - Dictionary containing the sentiment classification result
    """
    # Loading the sentiment analysis model
    sentiment_model = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    # Analyze sentiment of the given feedback
    sentiment_result = sentiment_model(feedback)
    return sentiment_result

# test_function_code --------------------

def test_analyze_customer_feedback():
    print("Testing started.")

    # Test case 1: Positive feedback
    positive_feedback = "Todo excelente, muy satisfecho con el servicio."
    assert analyze_customer_feedback(positive_feedback)[0]['label'] == 'POS', "Test case failed: Positive sentiment not detected."

    # Test case 2: Negative feedback
    negative_feedback = "Terrible experiencia, no recomiendo este producto."
    assert analyze_customer_feedback(negative_feedback)[0]['label'] == 'NEG', "Test case failed: Negative sentiment not detected."

    # Test case 3: Neutral feedback
    neutral_feedback = "El producto llegó a tiempo, pero esperaba más información del vendedor."
    assert analyze_customer_feedback(neutral_feedback)[0]['label'] == 'NEU', "Test case failed: Neutral sentiment not detected."

    print("Testing finished.")

# Running the test function
test_analyze_customer_feedback()