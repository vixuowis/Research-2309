# function_import --------------------

from transformers import pipeline

# function_code --------------------

def emotion_classification(user_message):
    """
    This function uses a pre-trained model from Hugging Face Transformers to classify the emotion in a given text.

    Args:
        user_message (str): The text message from the user.

    Returns:
        dict: The emotion classification result.

    Raises:
        OSError: If there is a problem with the disk quota.
    """
    
    # Load pre-trained model tokenizer
    model_name = 'distilbert-base-uncased'
    model = pipeline('text-classification', model=model_name)

    result = model(user_message)[0]
        
    return {
        "label": result["label"],  # emotion label as a string (e.g., 'sadness')
        "confidence": round(result["score"] * 100, 2)   # confidence of the prediction as a %
    }

# test_function_code --------------------

def test_emotion_classification():
    """
    This function tests the emotion_classification function with different test cases.
    """
    test_case_1 = 'I am feeling a bit down today.'
    test_case_2 = 'I am so happy!'
    test_case_3 = 'I am really angry at you.'

    assert isinstance(emotion_classification(test_case_1), list)
    assert isinstance(emotion_classification(test_case_2), list)
    assert isinstance(emotion_classification(test_case_3), list)

    return 'All Tests Passed'


# call_test_function_code --------------------

test_emotion_classification()