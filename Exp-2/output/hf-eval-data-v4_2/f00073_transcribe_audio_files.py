# requirements_file --------------------

!pip install -U huggingsound torch librosa datasets transformers

# function_import --------------------

from huggingsound import SpeechRecognitionModel

# function_code --------------------

def transcribe_audio_files(audio_paths):
    """Transcribe a list of audio files using a pre-trained ASR model.

    Args:
        audio_paths (list[str]): A list of paths to the audio files to be transcribed.

    Returns:
        list[str]: A list of transcriptions corresponding to the audio files.

    Raises:
        FileNotFoundError: If any audio file in the list does not exist.
    """
    model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-english')
    for path in audio_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"The audio file at {path} was not found.")
    return model.transcribe(audio_paths)

# test_function_code --------------------

def test_transcribe_audio_files():
    print("Testing started.")
    dataset = ['mock_audio1.wav', 'mock_audio2.wav']
    
    print("Testing case [1/1] started.")
    try:
        transcriptions = transcribe_audio_files(dataset)
        assert isinstance(transcriptions, list), f"Test case [1/1] failed: Expected list, got {type(transcriptions).__name__}"
    except FileNotFoundError as e:
        assert False, f"Test case [1/1] failed: {e}"
    print("Testing finished.")


# call_test_function_line --------------------

test_transcribe_audio_files()