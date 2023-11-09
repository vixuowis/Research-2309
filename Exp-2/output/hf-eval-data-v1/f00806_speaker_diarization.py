from pyannote.audio import Pipeline


def speaker_diarization(audio_file_path: str, access_token: str) -> None:
    """
    This function uses the pyannote.audio library to perform speaker diarization on an audio file.
    The diarization results are written to an RTTM file.

    Args:
        audio_file_path (str): The path to the audio file to be processed.
        access_token (str): The access token for the pyannote.audio API.

    Returns:
        None
    """
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1', use_auth_token=access_token)
    diarization = pipeline(audio_file_path)
    with open('audio.rttm', 'w') as rttm:
        diarization.write_rttm(rttm)