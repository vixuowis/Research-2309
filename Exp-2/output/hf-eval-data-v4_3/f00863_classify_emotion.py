# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_emotion(text):
    """Classify the emotion of the given text.

    Args:
        text (str): The text to be analyzed for emotion.

    Returns:
        dict: A dictionary containing the most likely emotion and its score.

    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')
    emotion_classifier = pipeline('sentiment-analysis', model='michellejieli/emotion_text_classifier')
    return emotion_classifier(text)

# test_function_code --------------------

def test_classify_emotion():
    print("Testing started.")
    # Test cases with example strings
    test_cases = [
        ("I'm feeling a bit down today.", 'sadness'),
        ("Wow! I never expected to win!", 'surprise'),
        ("This is the best day of my life!", 'joy')
    ]
    for i, (text, expected_emotion) in enumerate(test_cases):
        print(f"Testing case [{i+1}/{len(test_cases)}] started.")
        result = classify_emotion(text)
        assert result[0]['label'].lower() == expected_emotion, f"Test case [{i+1}/{len(test_cases)}] failed: Expected {expected_emotion}, got {result[0]['label']}."
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_emotion()