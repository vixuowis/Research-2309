# requirements_file --------------------

!pip install -U transformers torch tensorflow

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_emotion(text):
    """
    Classify the emotion of the given text using a pre-trained model.

    Args:
        text (str): A string containing the text to be classified.

    Returns:
        dict: A dictionary containing the classified emotion and its corresponding score.

    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input must be a string.')

    model_name = 'joeddav/distilbert-base-uncased-go-emotions-student'
    classifier = pipeline('text-classification', model=model_name)
    result = classifier(text)
    return result

# test_function_code --------------------

def test_classify_emotion():
    print("Testing started.")

    # Test case 1: Classify positive emotion
    print("Testing case [1/3] started.")
    positive_result = classify_emotion('I am so happy today!')
    assert positive_result[0]['label'] in ['joy', 'love', 'optimism'], f"Test case [1/3] failed: {positive_result}"

    # Test case 2: Classify negative emotion
    print("Testing case [2/3] started.")
    negative_result = classify_emotion('I am so sad today.')
    assert negative_result[0]['label'] in ['sadness', 'anger', 'fear'], f"Test case [2/3] failed: {negative_result}"

    # Test case 3: Handle non-string input
    print("Testing case [3/3] started.")
    try:
        classify_emotion(None)
        assert False, "Test case [3/3] failed: ValueError not raised for non-string input."
    except ValueError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_classify_emotion()