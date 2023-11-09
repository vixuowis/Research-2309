from pyannote.audio import Pipeline


def generate_transcript(audio_file):
    '''
    This function generates speaker diarization for a given audio file.
    
    Parameters:
    audio_file (str): Path to the audio file.
    
    Returns:
    diarization: Speaker diarization results.
    '''
    # Load the pre-trained speaker diarization model
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1')
    
    # Process the audio file with the loaded model
    diarization = pipeline(audio_file)
    
    return diarization