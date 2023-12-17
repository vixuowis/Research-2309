# requirements_file --------------------

!pip install -U huggingsound, torch, librosa, datasets, transformers

# function_import --------------------

from huggingsound import SpeechRecognitionModel

# function_code --------------------

def transcribe_audio_files(audio_paths):
    """
    Transcribe a list of audio file paths into Chinese text.

    Parameters:
        audio_paths (list): A list of strings, where each string is the file path to an audio file.

    Returns:
        list: A list containing the transcriptions of the audio files.
    """
    # Load the pretrained model from Hugging Face Transformers for Chinese speech recognition
    model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn')
    
    # Perform transcription on the list of audio file paths
    transcriptions = model.transcribe(audio_paths)
    return transcriptions


# test_function_code --------------------

def test_transcribe_audio_files():
    print("Testing transcribe_audio_files function.")
    # Example audio paths for testing
    audio_paths = ['audio_sample1.mp3', 'audio_sample2.wav']

    # Expected result format
    expected_output_format = [str]

    # Function call
    transcriptions = transcribe_audio_files(audio_paths)

    assert isinstance(transcriptions, list), f"The output should be a list, got {type(transcriptions)} instead."
    for transcription in transcriptions:
        assert isinstance(transcription, str), f"Each transcription should be a string, got {type(transcription)} instead."
    print("Passed all tests!")

# Run the test function
test_transcribe_audio_files()
