# function_import --------------------

from huggingsound import SpeechRecognitionModel

# function_code --------------------

def transcribe_arabic_speech(audio_paths):
    """
    Transcribe Arabic speech to text using a pre-trained Wav2Vec2 model.

    Args:
        audio_paths (list): A list of paths to audio files to be transcribed.

    Returns:
        list: A list of transcriptions for the provided audio files.
    """
    model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-arabic')
    transcriptions = model.transcribe(audio_paths)
    return transcriptions

# test_function_code --------------------

def test_transcribe_arabic_speech():
    """
    Test the transcribe_arabic_speech function with a sample audio file.
    """
    audio_paths = ['/path/to/sample.mp3']
    transcriptions = transcribe_arabic_speech(audio_paths)
    assert isinstance(transcriptions, list), 'The output should be a list.'
    assert len(transcriptions) > 0, 'The list should not be empty.'
    assert isinstance(transcriptions[0], str), 'Each item in the list should be a string.'

# call_test_function_code --------------------

test_transcribe_arabic_speech()