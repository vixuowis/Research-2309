# requirements_file --------------------

!pip install -U transformers librosa

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# function_code --------------------

def transcribe_podcast_audio(audio_file_path):
    # Load the processor and model
    processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')

    # Load and process the audio file
    audio_data, sampling_rate = librosa.load(audio_file_path, sr=None)
    input_features = processor(audio_data, sampling_rate=sampling_rate, return_tensors='pt').input_features

    # Generate transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    # Return the transcription
    return transcription[0] if transcription else ''

# test_function_code --------------------

def test_transcribe_podcast_audio():
    print("Testing started.")
    test_audio_file = 'test_audio.wav' # Replace with a valid test audio file

    # Test case: Transcribing a sample audio file
    print("Testing transcription started.")
    transcription = transcribe_podcast_audio(test_audio_file)
    assert transcription, f"Test failed: Transcription is empty"
    print("Test passed: Transcription obtained")

    print("Testing finished.")

# Run the test function
test_transcribe_podcast_audio()