from pyannote.audio import Pipeline


def speaker_diarization(audio_file: str, access_token: str) -> None:
    """
    Function to perform speaker diarization on an audio file using a pretrained model from pyannote.audio.

    Args:
        audio_file (str): Path to the audio file to be processed.
        access_token (str): Access token for the pretrained model.

    Returns:
        None. The function writes the output to an RTTM file.
    """
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1', use_auth_token=access_token)
    diarization = pipeline(audio_file)
    with open(f'{audio_file}.rttm', 'w') as rttm:
        diarization.write_rttm(rttm)