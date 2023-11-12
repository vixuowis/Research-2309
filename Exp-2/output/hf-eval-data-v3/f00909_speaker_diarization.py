# function_import --------------------

from pyannote.audio import Pipeline
import os

# function_code --------------------

def speaker_diarization(audio_file: str, access_token: str) -> None:
    """
    This function uses the pyannote.audio library to perform speaker diarization on an audio file.
    The output is saved in an RTTM (Rich Text Time-Marked) format.

    Args:
        audio_file (str): The path to the audio file to be processed.
        access_token (str): The access token for the pretrained model.

    Returns:
        None
    """
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1', use_auth_token=access_token)
    diarization = pipeline(audio_file)
    with open(f'{audio_file}.rttm', 'w') as rttm:
        diarization.write_rttm(rttm)

# test_function_code --------------------

def test_speaker_diarization():
    """
    This function tests the speaker_diarization function.
    """
    # Test case 1: Normal case
    speaker_diarization('test_audio.wav', 'test_token')
    assert os.path.exists('test_audio.wav.rttm')
    # Test case 2: File does not exist
    try:
        speaker_diarization('non_existent_file.wav', 'test_token')
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Expected FileNotFoundError')
    # Test case 3: Invalid token
    try:
        speaker_diarization('test_audio.wav', 'invalid_token')
    except Exception:
        pass
    else:
        raise AssertionError('Expected Exception')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_speaker_diarization()