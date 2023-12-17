# requirements_file --------------------

!pip install -U huggingsound torch librosa datasets transformers

# function_import --------------------

from huggingsound import SpeechRecognitionModel

# function_code --------------------

def transcribe_arabic_audio(audio_paths):
    """
    Transcribe a list of Arabic audio files to text.

    Parameters:
        audio_paths (list of str): Paths to the audio files to be transcribed.

    Returns:
        dict: A dictionary mapping audio file paths to their transcriptions.
    """
    model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-arabic')
    transcriptions = model.transcribe(audio_paths)
    return {audio_path: transcriptions[i] for i, audio_path in enumerate(audio_paths)}

# test_function_code --------------------

def test_transcribe_arabic_audio():
    print("Testing started.")
    audio_paths = ['./audio_sample1.mp3', './audio_sample2.wav']

    # Mocked transcriptions
    mocked_transcriptions = {'./audio_sample1.mp3': 'مرحبا كيف حالك', './audio_sample2.wav': 'أهلا وسهلا'}

    print("Testing transcribe_arabic_audio started.")
    transcriptions = transcribe_arabic_audio(audio_paths)
    assert transcriptions == mocked_transcriptions, f"Test failed: Expected {mocked_transcriptions}, got {transcriptions}"
    print("Testing finished.")

# Run the test
test_transcribe_arabic_audio()