# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_emotion(text):
    """
    Classify the type of emotion in a given movie review text.

    Args:
        text (str): The movie review text to be classified.

    Returns:
        dict: A dictionary containing the predicted emotion and its score.

    Raises:
        ValueError: If the input text is empty.
    """
    if not text:
        raise ValueError('Input text cannot be empty')

    # Load the sentiment analysis pipeline with the specified model
    classifier = pipeline('sentiment-analysis', model='michellejieli/emotion_text_classifier')

    # Return the emotion classification result
    return classifier(text)

# test_function_code --------------------

def test_classify_emotion():
    print("Testing started.")

    # Testing case 1: Positive sentiment test
    print("Testing case [1/3] started.")
    positive_text = "What a fantastic movie! It was so captivating."
    positive_result = classify_emotion(positive_text)
    assert positive_result[0]['label'] == 'POSITIVE', f"Test case [1/3] failed: {positive_result}"

    # Testing case 2: Negative sentiment test
    print("Testing case [2/3] started.")
    negative_text = "This movie was terrible and boring."
    negative_result = classify_emotion(negative_text)
    assert negative_result[0]['label'] == 'NEGATIVE', f"Test case [2/3] failed: {negative_result}"

    # Testing case 3: Empty string test
    print("Testing case [3/3] started.")
    try:
        _ = classify_emotion('')
        assert False, "Test case [3/3] failed: No error was raised for empty input"
    except ValueError as e:
        assert str(e) == 'Input text cannot be empty', f"Test case [3/3] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_classify_emotion()