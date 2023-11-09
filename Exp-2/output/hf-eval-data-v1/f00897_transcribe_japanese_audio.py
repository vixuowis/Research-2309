from huggingsound import SpeechRecognitionModel


def transcribe_japanese_audio(audio_paths):
    """
    Transcribes Japanese audio files using the 'jonatasgrosman/wav2vec2-large-xlsr-53-japanese' model from Hugging Face Transformers.

    Args:
        audio_paths (list): A list of paths to the audio files to be transcribed.

    Returns:
        list: A list of transcriptions for the provided audio files.
    """
    model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-japanese')
    transcriptions = model.transcribe(audio_paths)
    return transcriptions