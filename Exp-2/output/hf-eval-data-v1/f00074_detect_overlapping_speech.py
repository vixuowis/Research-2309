from pyannote.audio import Pipeline


def detect_overlapping_speech(audio_file, access_token):
    """
    This function detects overlapping speech in an audio file using the pyannote.audio framework.
    The model detects when two or more speakers are active in an audio file.
    
    Parameters:
    audio_file (str): The path to the audio file.
    access_token (str): The access token for the pretrained model.
    
    Returns:
    list: A list of tuples where each tuple represents a segment of overlapping speech. Each tuple contains the start and end times of the segment.
    """
    pipeline = Pipeline.from_pretrained('pyannote/overlapped-speech-detection', use_auth_token=access_token)
    output = pipeline(audio_file)
    overlapping_speech_segments = []
    for speech in output.get_timeline().support():
        overlapping_speech_segments.append((speech.start, speech.end))
    return overlapping_speech_segments