# requirements_file --------------------

!pip install -U pyannote.audio

# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def perform_speaker_diarization(audio_file_path):
    # Load the pre-trained speaker diarization pipeline
    pipeline = Pipeline.from_pretrained('philschmid/pyannote-speaker-diarization-endpoint')

    # Run the diarization on the provided audio file
    diarization_result = pipeline(audio_file_path)

    # Extract and return speaker activity segments
    speaker_segments = diarization_result.get_timeline()
    return speaker_segments

# test_function_code --------------------

def test_perform_speaker_diarization():
    print("Testing perform_speaker_diarization...")
    sample_audio = 'example.wav'  # Replace with a path to a real audio file for actual testing

    # Test case 1: Check if the function returns a non-empty result
    result = perform_speaker_diarization(sample_audio)
    assert len(result) > 0, "Test case failed: The result should not be empty."

    print("Test completed successfully.")

# Run the test function
test_perform_speaker_diarization()