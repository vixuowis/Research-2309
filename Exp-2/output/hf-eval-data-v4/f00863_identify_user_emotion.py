# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def identify_user_emotion(text):
    # Initialize the sentiment analysis pipeline with the specific model
    emotion_classifier = pipeline('sentiment-analysis', model='michellejieli/emotion_text_classifier')
    # Process the input text and predict the emotion
    emotion_result = emotion_classifier(text)
    # Return the result
    return emotion_result

# test_function_code --------------------

def test_identify_user_emotion():
    print("Testing started.")

    sample_texts = [
        "I'm feeling a bit down today.", # Expected emotion: Sadness
        "What a marvelous day!", # Expected emotion: Joy
        "This is disgusting!", # Expected emotion: Disgust
    ]

    for i, text in enumerate(sample_texts):
        print(f"Testing case [{i+1}/{len(sample_texts)}] started.")
        result = identify_user_emotion(text)
        assert len(result) > 0, f"Test case [{i+1}/{len(sample_texts)}] failed: No result returned"
        # We expect the first (highest-ranking) result to be the predicted emotion
        assert type(result[0]) is dict, f"Test case [{i+1}/{len(sample_texts)}] failed: The result is not a dictionary"
        assert 'label' in result[0], f"Test case [{i+1}/{len(sample_texts)}] failed: 'label' not in result"
        print(f"Testing case [{i+1}/{len(sample_texts)}] completed.")
    
    print("Testing finished.")

# Run the test function
test_identify_user_emotion()