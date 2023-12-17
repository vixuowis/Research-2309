# requirements_file --------------------

import subprocess

requirements = ["huggingsound", "torch", "librosa", "datasets", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from huggingsound import SpeechRecognitionModel

# function_code --------------------

def arabic_speech_to_text(audio_paths):
    """
    Convert a list of audio file paths to their corresponding text transcriptions in Arabic.

    Args:
        audio_paths (List[str]): A list of strings containing the file paths to audio files.

    Returns:
        List[str]: Returns a list of transcriptions in Arabic for the corresponding audio files.

    Raises:
        FileNotFoundError: If any of the audio file paths does not exist.
        Exception: If the transcription fails due to model or audio processing errors.
    """
    model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-arabic')
    transcriptions = model.transcribe(audio_paths)
    return transcriptions

# test_function_code --------------------

def test_arabic_speech_to_text():
    print("Testing started.")

    # Test with a list of audio file paths
    print("Testing case [1/2] started.")
    audio_paths = ['audio_file_1.mp3', 'audio_file_2.wav']
    transcriptions = arabic_speech_to_text(audio_paths)
    assert len(transcriptions) == len(audio_paths), f"Test case [1/2] failed: Expected {len(audio_paths)} transcriptions, got {len(transcriptions)}"

    # Test with non-existent audio file paths
    print("Testing case [2/2] started.")
    non_existent_audio_paths = ['non_existent_audio_file_1.mp3']
    try:
        arabic_speech_to_text(non_existent_audio_paths)
        assert False, "Test case [2/2] failed: FileNotFoundError was not raised for non-existent audio files"
    except FileNotFoundError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_arabic_speech_to_text()