# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_spoken_digits(audio_file_path):
    """Classify spoken digits in an audio file using a pre-trained model.

    Args:
        audio_file_path (str): The path to the audio file to be analyzed.

    Returns:
        list: A list of detected digits with their confidence scores.

    Raises:
        FileNotFoundError: If the audio file is not found at the specified path.
        ValueError: If the audio file is not in a suitable format.
    """
    # Initialize the digit audio classification pipeline with pre-trained model
    digit_classifier = pipeline('audio-classification', model='mazkooleg/0-9up-data2vec-audio-base-960h-ft')

    # Perform digit classification on the provided audio file
    try:
        digits_detected = digit_classifier(audio_file_path)
    except FileNotFoundError as e:
        raise FileNotFoundError("Audio file not found.") from e
    except Exception as e:
        raise ValueError("Audio processing error.") from e

    return digits_detected

# test_function_code --------------------

def test_classify_spoken_digits(audio_file_path, expected_output):
    print("Testing started.")
    try:
        # Classify spoken digits
        result = classify_spoken_digits(audio_file_path)
        assert result == expected_output, f"Test failed: Expected {expected_output}, got {result}"
    except Exception as e:
        print(f"Test failed with exception: {e}")
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_spoken_digits('sample_audio.wav', expected_digits)