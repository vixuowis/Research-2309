from huggingsound import SpeechRecognitionModel


def transcribe_arabic_speech(audio_paths):
    """
    Transcribes Arabic speech from audio files using a pre-trained Wav2Vec2 model.

    Args:
        audio_paths (list): A list of paths to audio files to be transcribed.

    Returns:
        A list of transcriptions corresponding to the input audio files.
    """
    model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-arabic')
    transcriptions = model.transcribe(audio_paths)
    return transcriptions