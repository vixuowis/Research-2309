# requirements_file --------------------

!pip install -U transformers torch tensorflow

# function_import --------------------

from transformers import pipeline

# function_code --------------------



## Function: classify_emotion

This function classifies the emotion expressed in the given text using a pre-trained model.

Args:
    text (str): The text to be classified.

Returns:
    dict: The classification result containing the label and score.

def classify_emotion(text):
    nlp = pipeline('text-classification', model='joeddav/distilbert-base-uncased-go-emotions-student')
    return nlp(text)


# test_function_code --------------------



def test_classify_emotion():
    print("Testing started.")

    # Test case 1: Check if the function returns a dictionary
    print("Testing case [1/3] started.")
    result = classify_emotion('I am so happy today!')
    assert isinstance(result, dict), f"Test case [1/3] failed: Expected result to be a dictionary, got {type(result)}"

    # Test case 2: Check if the 'label' key is in the result
    print("Testing case [2/3] started.")
    assert 'label' in result, f"Test case [2/3] failed: 'label' key not found in result"

    # Test case 3: Check if the function handles empty string
    print("Testing case [3/3] started.")
    result = classify_emotion('')
    assert isinstance(result, dict), f"Test case [3/3] failed: Expected result to be a dictionary, got {type(result)}"
    print("Testing finished.")

# Run the test function
test_classify_emotion()
