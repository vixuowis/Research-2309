from huggingsound import SpeechRecognitionModel


def transcribe_russian_lessons(audio_paths):
    """
    This function transcribes Russian audio lessons into text using the 'jonatasgrosman/wav2vec2-large-xlsr-53-russian' model.
    
    Parameters:
    audio_paths (list): A list of paths to the audio files to be transcribed.
    
    Returns:
    list: A list of transcriptions for each audio file.
    """
    # Create an instance of the SpeechRecognitionModel class and load the 'jonatasgrosman/wav2vec2-large-xlsr-53-russian' model
    model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-russian')
    
    # Transcribe the audio files
    transcriptions = model.transcribe(audio_paths)
    
    return transcriptions