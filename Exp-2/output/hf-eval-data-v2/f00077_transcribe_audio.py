# function_import --------------------

from huggingsound import SpeechRecognitionModel

# function_code --------------------

def transcribe_audio(audio_paths):
    """
    Transcribe Chinese language audio files into text.

    Args:
        audio_paths (list): A list of paths to the audio files.

    Returns:
        list: A list of transcriptions for each audio file.
    """
    model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn')
    transcriptions = model.transcribe(audio_paths)
    return transcriptions

# test_function_code --------------------

def test_transcribe_audio():
    """
    Test the transcribe_audio function.
    """
    audio_paths = ['/path/to/test_file.mp3', '/path/to/test_file.wav']
    transcriptions = transcribe_audio(audio_paths)
    assert isinstance(transcriptions, list), 'The result should be a list.'
    assert len(transcriptions) == len(audio_paths), 'The number of transcriptions should be equal to the number of audio files.'

# call_test_function_code --------------------

test_transcribe_audio()