# requirements_file --------------------

import subprocess

requirements = ["pyannote.audio"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from pyannote.audio.core.inference import Inference

# function_code --------------------

def extract_voice_activity(audio_file, device='cuda'):
    """Extract voice activity segments from an audio file.

    Args:
        audio_file (str): Path to the audio file to process.
        device (str): The device to perform inference on. Default is 'cuda'.

    Returns:
        List[Tuple[float, float]]: List of tuples representing the start and end times of voice activity segments.

    Raises:
        ValueError: If the audio file is not found or not accessible.
    """
    model = Inference('julien-c/voice-activity-detection', device=device)
    result = model({'audio': audio_file})
    return [(segment.start, segment.end) for segment in result.iter_segments()]


# test_function_code --------------------

def test_extract_voice_activity():
    print("Testing started.")
    audio_file = 'example.wav'

    # Test case 1: Check if function returns a list
    print("Testing case [1/2] started.")
    segments = extract_voice_activity(audio_file)
    assert isinstance(segments, list), f"Test case [1/2] failed: Expected result type list, got {type(segments)}"

    # Test case 2: Check if each segment is a tuple with two float values
    print("Testing case [2/2] started.")
    for segment in segments:
        assert isinstance(segment, tuple) and len(segment) == 2 and all(isinstance(t, float) for t in segment), f"Test case [2/2] failed: Segment {segment} is not a valid tuple of floats."
    print("Testing finished.")


# call_test_function_line --------------------

test_extract_voice_activity()