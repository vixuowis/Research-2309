# function_import --------------------

import os
from pyannote.audio import Pipeline

# function_code --------------------

def speaker_diarization(audio_file_path: str, access_token: str) -> None:
    """
    This function uses the pyannote.audio library to perform speaker diarization on an audio file.
    The diarization results are written to an RTTM file.

    Args:
        audio_file_path (str): The path to the audio file.
        access_token (str): The access token for the pretrained model.

    Returns:
        None
    """
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1', use_auth_token=access_token)
    diarization = pipeline(audio_file_path)
    with open('audio.rttm', 'w') as rttm:
        diarization.write_rttm(rttm)

# test_function_code --------------------

def test_speaker_diarization():
    """
    This function tests the speaker_diarization function.
    """
    # Test case 1: Valid audio file and access token
    speaker_diarization('path/to/valid/audio.wav', 'valid_access_token')
    assert os.path.exists('audio.rttm')
    # Test case 2: Invalid audio file
    try:
        speaker_diarization('path/to/invalid/audio.wav', 'valid_access_token')
    except Exception as e:
        assert isinstance(e, FileNotFoundError)
    # Test case 3: Invalid access token
    try:
        speaker_diarization('path/to/valid/audio.wav', 'invalid_access_token')
    except Exception as e:
        assert isinstance(e, ValueError)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_speaker_diarization()