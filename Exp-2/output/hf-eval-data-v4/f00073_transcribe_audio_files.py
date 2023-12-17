# requirements_file --------------------

!pip install -U huggingsound torch librosa datasets transformers

# function_import --------------------

from huggingsound import SpeechRecognitionModel

# function_code --------------------

def transcribe_audio_files(audio_paths):
    """
    Transcribes a list of audio file paths using a pre-trained ASR model.
    
    :param audio_paths: List of file paths to the audio files.
    :return: List of transcriptions for each audio file.
    """
    model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-english')
    transcriptions = model.transcribe(audio_paths)
    return transcriptions

# test_function_code --------------------

def test_transcribe_audio_files():
    print("Testing transcribe_audio_files function started.")
    
    audio_paths = ['test_audio1.wav', 'test_audio2.wav', 'test_audio3.wav']
    expected_transcriptions = ['Sample text 1', 'Sample text 2', 'Sample text 3']

    # Test transcribe_audio_files function with several audio files
    transcriptions = transcribe_audio_files(audio_paths)
    for i, transcription in enumerate(transcriptions):
        assert transcription == expected_transcriptions[i], f"Test case [{i+1}] failed: Expected '{{expected_transcriptions[i]}}', but got '{{transcription}}'"
    
    print("Testing transcribe_audio_files function finished successfully.")

# Run the test function
test_transcribe_audio_files()