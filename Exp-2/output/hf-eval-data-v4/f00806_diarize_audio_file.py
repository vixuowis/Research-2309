# requirements_file --------------------

!pip install -U pyannote.audio==2.1.1

# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def diarize_audio_file(audio_file_path, access_token):
    """
    Apply speaker diarization on an audio file using pyannote.audio pre-trained model.

    :param audio_file_path: Path to the audio file to be analyzed.
    :param access_token: Token to access the pre-trained model.
    :return: Diarization object with speaker segmentation information.
    """
    # Load the pre-trained pipeline for diarization
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1', use_auth_token=access_token)

    # Run diarization on the provided audio file
    diarization = pipeline(audio_file_path)
    return diarization

# test_function_code --------------------

def test_diarize_audio_file():
    print("Testing diarize_audio_file function.")
    
    # Assuming 'demo_audio.wav' is a valid audio file for testing
    test_audio_path = 'demo_audio.wav'
    test_access_token = 'test_token'
    
    # Run the diarization function
    diarization_result = diarize_audio_file(test_audio_path, test_access_token)
    
    # Check if the diarization result is not None
    assert diarization_result is not None, "The diarization result should not be None"

    print("All tests passed.")

# Execute the test
test_diarize_audio_file()