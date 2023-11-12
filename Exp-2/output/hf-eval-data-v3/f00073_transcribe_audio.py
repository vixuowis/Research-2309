# function_import --------------------

from huggingsound import SpeechRecognitionModel

# function_code --------------------

def transcribe_audio(audio_paths):
    """
    Transcribe audio files into text using a pre-trained ASR model.

    Args:
        audio_paths (list): A list of paths to the audio files.

    Returns:
        list: A list of transcriptions for each audio file.
    """
    model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-english')
    transcriptions = model.transcribe(audio_paths)
    return transcriptions

# test_function_code --------------------

def test_transcribe_audio():
    """
    Test the transcribe_audio function with some example audio files.
    """
    audio_paths = ['path/to/first_audio.mp3', 'path/to/second_audio.wav']
    transcriptions = transcribe_audio(audio_paths)
    assert isinstance(transcriptions, list), 'The output should be a list.'
    assert len(transcriptions) == len(audio_paths), 'The number of transcriptions should match the number of audio files.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_transcribe_audio()