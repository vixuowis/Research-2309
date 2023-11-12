# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf

# function_code --------------------

def transcribe_audio(audio_file_path):
    """
    Transcribe audio file into text using Hugging Face's Whisper ASR model.

    Args:
        audio_file_path (str): Path to the audio file to be transcribed.

    Returns:
        str: Transcribed text from the audio file.

    Raises:
        FileNotFoundError: If the audio file is not found at the provided path.
        Exception: If any error occurs during the transcription process.
    """
    try:
        # Load the audio file
        audio_data, sampling_rate = sf.read(audio_file_path)

        # Load the Whisper ASR processor and model
        processor = WhisperProcessor.from_pretrained('openai/whisper-tiny.en')
        model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny.en')

        # Process the audio data and generate predictions
        input_features = processor(audio_data, sampling_rate=sampling_rate, return_tensors='pt').input_features
        predicted_ids = model.generate(input_features)

        # Decode the predictions into text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return transcription
    except FileNotFoundError as fnf_error:
        print(f'Error: {fnf_error}')
        raise
    except Exception as e:
        print(f'Error: {e}')
        raise

# test_function_code --------------------

def test_transcribe_audio():
    """
    Test the transcribe_audio function with a sample audio file.
    """
    # Test with a sample audio file
    try:
        transcription = transcribe_audio('sample.wav')
        assert isinstance(transcription, str), 'The transcription should be a string.'
        print('Test passed.')
    except Exception as e:
        print(f'Test failed: {e}')

# call_test_function_code --------------------

test_transcribe_audio()