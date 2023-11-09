# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# function_code --------------------

def transcribe_audio(audio_file_path):
    """
    Transcribe an audio file using the Whisper ASR model from Hugging Face Transformers.

    Args:
        audio_file_path (str): The path to the audio file to transcribe.

    Returns:
        str: The transcription of the audio file.
    """
    # Load the Whisper ASR model and processor
    processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')

    # Load the audio file
    audio_data, sampling_rate = librosa.load(audio_file_path, sr=None)

    # Process the audio file and generate the input features
    input_features = processor(audio_data, sampling_rate=sampling_rate, return_tensors='pt').input_features

    # Generate the predicted IDs
    predicted_ids = model.generate(input_features)

    # Decode the predicted IDs to get the transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription

# test_function_code --------------------

def test_transcribe_audio():
    """
    Test the transcribe_audio function.
    """
    # Define a test audio file path
    test_audio_file_path = 'path/to/test/audio/file.wav'

    # Call the transcribe_audio function
    transcription = transcribe_audio(test_audio_file_path)

    # Assert that the transcription is not empty
    assert transcription != ''

# call_test_function_code --------------------

test_transcribe_audio()