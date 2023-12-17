# requirements_file --------------------

!pip install -U transformers==4.26.1 torch==1.11.0 datasets==2.10.0 tokenizers==0.12.1

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_spoken_digit(audio_sample_path):
    """
    Classify a spoken digit from an audio file using a pre-trained Hugging Face Transformers model.

    Parameters:
    audio_sample_path (str): The file path to the audio sample to be classified.

    Returns:
    int: The digit classification result.
    """
    spoken_digit_classifier = pipeline('audio-classification', model='MIT/ast-finetuned-speech-commands-v2')
    prediction = spoken_digit_classifier(audio_sample_path)
    # Assuming prediction is a list and the digit is the first item of the predicted label's id
    return int(prediction[0]['label'].split('_')[-1])

# test_function_code --------------------

def test_classify_spoken_digit():
    print("Testing started.")
    # Assuming test files are named as "digit0.wav", "digit1.wav", ..., "digit9.wav"
    for digit in range(10):
        audio_sample_path = f"test_data/digit{digit}.wav"
        print(f"Testing case [{digit+1}/10] started.")
        predicted_digit = classify_spoken_digit(audio_sample_path)
        assert predicted_digit == digit, f"Test case [{digit+1}/10] failed: Expected {digit}, got {predicted_digit}"
        print(f"Testing case [{digit+1}/10] successful.")

    print("Testing finished.")

# Run the test function
test_classify_spoken_digit()