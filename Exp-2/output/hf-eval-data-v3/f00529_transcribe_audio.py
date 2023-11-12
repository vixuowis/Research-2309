# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# function_code --------------------

def transcribe_audio(audio_file_path):
    """
    Transcribe an audio file using the Whisper ASR model from Hugging Face Transformers.

    Args:
        audio_file_path (str): Path to the audio file to transcribe.

    Returns:
        str: The transcription of the audio file.
    """
    processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')

    audio_data, sampling_rate = librosa.load(audio_file_path, sr=None)
    input_features = processor(audio_data, sampling_rate=sampling_rate, return_tensors='pt').input_features

    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription

# test_function_code --------------------

def test_transcribe_audio():
    """
    Test the transcribe_audio function.
    """
    # Test with a short audio file
    transcription = transcribe_audio('path/to/short/audio/file.wav')
    assert isinstance(transcription, str), 'The transcription should be a string.'

    # Test with a longer audio file
    transcription = transcribe_audio('path/to/long/audio/file.wav')
    assert isinstance(transcription, str), 'The transcription should be a string.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_transcribe_audio()