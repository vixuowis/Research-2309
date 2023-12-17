# requirements_file --------------------

!pip install -U pyannote.audio

# function_import --------------------

from typing import List, Tuple
from pyannote.audio import Pipeline

# function_code --------------------

def detect_overlapping_speech(audio_file_path: str, access_token: str) -> List[Tuple[float, float]]:
    """
    Detect overlapping speech segments in an audio file using pyannote.audio.

    :param audio_file_path: Path to the audio file to be processed.
    :param access_token: Access token for using the pre-trained models from pyannote.audio.
    :return: A list of tuples, each containing the start and end times of an overlapping speech segment.
    """
    # Load the pre-trained overlapped-speech-detection model
    pipeline = Pipeline.from_pretrained('pyannote/overlapped-speech-detection', use_auth_token=access_token)
    
    # Process the audio file and get the timeline of overlapping speech
    output = pipeline(audio_file_path)
    
    # Extract start and end times from the overlapping speech segments
    overlapping_segments = [(speech.start, speech.end) for speech in output.get_timeline().support()]
    return overlapping_segments

# test_function_code --------------------

def test_detect_overlapping_speech():
    print("Testing started.")
    
    # Provide valid audio file path and access token for testing
    audio_file_path = "test_audio.wav"  # Replace with a valid audio file path
    access_token = "your_access_token_here"  # Replace with a valid access token

    # Testing the function with an actual audio file
    print("Testing case [1/1] started.")
    overlapping_segments = detect_overlapping_speech(audio_file_path, access_token)
    
    # We cannot assert a specific result since it depends on the audio
    # Instead, we'll check if the function returns a list
    assert isinstance(overlapping_segments, list), f"Test case [1/1] failed: Function should return a list, not {type(overlapping_segments)}"
    
    # Optionally, you may print the overlapping segments for manual inspection
    for start_time, end_time in overlapping_segments:
        print(f"Overlapping speech detected from {start_time:.2f} to {end_time:.2f} seconds.")
    
    print("Testing finished.")

# Run the test function (This will likely raise an error without actual access token and audio file)
test_detect_overlapping_speech()