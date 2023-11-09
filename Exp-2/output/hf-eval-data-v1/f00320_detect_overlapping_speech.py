from pyannote.audio import Pipeline


def detect_overlapping_speech(audio_file: str, access_token: str):
    """
    This function detects overlapping speech in an audio file using the pyannote.audio framework.
    The model detects when two or more speakers are active in an audio file.

    Parameters:
    audio_file (str): The path to the audio file.
    access_token (str): The access token for the pyannote.audio API.

    Returns:
    list: A list of tuples, each containing the start and end times of overlapping speech segments.
    """
    # Load the pre-trained model
    pipeline = Pipeline.from_pretrained('pyannote/overlapped-speech-detection', use_auth_token=access_token)

    # Process the audio file
    output = pipeline(audio_file)

    # Initialize an empty list to store the overlapping speech segments
    overlapping_speech_segments = []

    # Iterate through the overlapping speech periods and extract the start and end times of each segment
    for speech in output.get_timeline().support():
        start_time, end_time = speech.start, speech.end
        overlapping_speech_segments.append((start_time, end_time))

    return overlapping_speech_segments