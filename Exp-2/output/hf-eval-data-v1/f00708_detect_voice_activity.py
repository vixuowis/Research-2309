from pyannote.audio import Pipeline


def detect_voice_activity(audio_file):
    """
    This function uses the pretrained voice activity detection pipeline from pyannote.audio to detect active speech in an audio file.
    
    Parameters:
    audio_file (str): Path to the audio file (.wav format).
    
    Returns:
    list: A list of tuples where each tuple represents the start and end times of an active speech segment.
    """
    # Load the pretrained pipeline
    pipeline = Pipeline.from_pretrained('pyannote/voice-activity-detection')
    
    # Apply the pipeline on the input audio file to get a segmentation output
    output = pipeline(audio_file)
    
    # Initialize an empty list to store the active speech segments
    active_speech_segments = []
    
    # Iterate through the output's timeline to identify the start and end times of each active speech segment
    for speech in output.get_timeline().support():
        # Active speech between speech.start and speech.end
        active_speech_segments.append((speech.start, speech.end))
    
    return active_speech_segments