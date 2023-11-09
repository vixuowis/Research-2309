# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def speaker_diarization(audio_file_path: str, access_token: str) -> None:
    """
    This function provides an automatic speaker diarization pipeline using the pyannote.audio framework.
    It processes audio files and outputs speaker diarization results in RTTM format.
    
    Args:
        audio_file_path (str): The path to the audio file to be processed.
        access_token (str): The access token for the pretrained model.
    
    Returns:
        None. The function writes the diarization results to an RTTM file.
    """
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1', use_auth_token=access_token)
    diarization = pipeline(audio_file_path)
    with open('audio.rttm', 'w') as rttm:
        diarization.write_rttm(rttm)

# test_function_code --------------------

def test_speaker_diarization():
    """
    This function tests the speaker_diarization function.
    It uses a sample audio file and checks if the output RTTM file is created.
    """
    speaker_diarization('sample_audio.wav', 'sample_access_token')
    assert os.path.exists('audio.rttm')

# call_test_function_code --------------------

test_speaker_diarization()