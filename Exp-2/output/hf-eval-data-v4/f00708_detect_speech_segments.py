# requirements_file --------------------

!pip install -U pyannote.audio==2.1 soundfile

# function_import --------------------

from pyannote.audio import Pipeline
import soundfile as sf

# function_code --------------------

def detect_speech_segments(audio_file_path):
    """
    Detect speech segments in an audio file using a pretrained voice activity detection model.

    Parameters:
    audio_file_path (str): The file path of the .wav audio file to be analyzed.

    Returns:
    list of tuples: A list where each tuple contains the start and end times of detected speech segments.
    """
    try:
        # Load the pretrained model
        pipeline = Pipeline.from_pretrained('pyannote/voice-activity-detection')

        # Process the input audio file
        output = pipeline(audio_file_path)

        # Extract the speech segments from the output
        speech_segments = [(speech.start, speech.end) for speech in output.get_timeline().support()]

        return speech_segments
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# test_function_code --------------------

def test_detect_speech_segments():
    print("Testing started.")
    # No need for a dataset, testing with a local audio file

    # Test case 1: Valid audio file
    print("Testing case [1/1] started.")
    speech_segments = detect_speech_segments("valid_audio.wav")
    assert speech_segments, f"Test case [1/1] failed: No speech segments detected."
    print("Testing case [1/1] passed.")
    print("Testing finished.")