# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_user_emotion(user_response_text):
    """
    Detect the emotion of a user response using emotion classification model.

    Args:
        user_response_text (str): The response text from the user.

    Returns:
        dict: A dictionary containing the detected emotion and the confidence score.

    Raises:
        ValueError: If the user_response_text is not a string or is empty.
    """
    if not user_response_text or not isinstance(user_response_text, str):
        raise ValueError('The user response must be a non-empty string.')

    emotion_detector = pipeline('sentiment-analysis', model='michellejieli/emotion_text_classifier')
    result = emotion_detector(user_response_text)[0]
    return {'emotion': result['label'], 'score': result['score']}

# test_function_code --------------------

def test_detect_user_emotion():
    print('Testing started.')
    # Here we use predefined responses for testing as we do not have a real user
    test_responses = [
        ('I am very happy today!', 'joy'),
        ('I am so scared of spiders.', 'fear'),
        ('I hate waiting in long lines.', 'anger')
    ]
    
    for i, (response, expected_emotion) in enumerate(test_responses, start=1):
        print(f'Testing case [{i}/{len(test_responses)}] started.')
        emotion_data = detect_user_emotion(response)
        assert emotion_data['emotion'].lower() == expected_emotion, f'Test case [{i}/{len(test_responses)}] failed: Expected emotion {expected_emotion}, but got {emotion_data['emotion']}.'
    
    print('Testing finished.')

# call_test_function_line --------------------

test_detect_user_emotion()