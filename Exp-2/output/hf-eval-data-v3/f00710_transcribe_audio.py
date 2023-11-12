# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf

# function_code --------------------

def transcribe_audio(audio_file_path):
    '''
    Transcribe an audio file using the Whisper ASR model from Hugging Face Transformers.

    Args:
        audio_file_path (str): The path to the audio file to transcribe.

    Returns:
        str: The transcription of the audio file.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        Exception: If any error occurs during the transcription process.
    '''
    try:
        # Load the audio file
        audio_data, sampling_rate = sf.read(audio_file_path)

        # Load the Whisper ASR processor and model
        processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
        model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')

        # Preprocess the audio data and generate input features
        input_features = processor(audio_data, sampling_rate=sampling_rate, return_tensors='pt').input_features

        # Generate the predicted transcription IDs
        predicted_ids = model.generate(input_features)

        # Decode the transcription
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
    '''
    Test the transcribe_audio function.
    '''
    # Test with a valid audio file
    transcription = transcribe_audio('valid_audio_file.wav')
    assert isinstance(transcription, str), 'The transcription should be a string.'

    # Test with a non-existent audio file
    try:
        transcription = transcribe_audio('non_existent_file.wav')
    except FileNotFoundError:
        pass
    else:
        assert False, 'A FileNotFoundError should have been raised.'

    # Test with an invalid audio file
    try:
        transcription = transcribe_audio('invalid_audio_file.wav')
    except Exception:
        pass
    else:
        assert False, 'An exception should have been raised.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_transcribe_audio()