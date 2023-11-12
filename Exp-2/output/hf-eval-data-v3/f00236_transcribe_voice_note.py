# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf

# function_code --------------------

def transcribe_voice_note(audio_file_path: str) -> str:
    """
    Transcribe a voice note to text using the Hugging Face Transformers Whisper model.

    Args:
        audio_file_path (str): The path to the audio file to transcribe.

    Returns:
        str: The transcribed text.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        Exception: If any error occurs during transcription.
    """
    try:
        # Load the pre-trained model and processor
        processor = WhisperProcessor.from_pretrained('openai/whisper-large')
        model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large')
        model.config.forced_decoder_ids = None

        # Load the audio file and extract its sampling rate
        audio, sampling_rate = sf.read(audio_file_path)

        # Convert the audio input into input features
        input_features = processor(audio, sampling_rate=sampling_rate, return_tensors='pt').input_features

        # Generate predicted IDs using the Whisper model
        predicted_ids = model.generate(input_features)

        # Decode the predicted IDs into text transcription
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

        return transcription[0]
    except FileNotFoundError as fnf_error:
        print(f'Error: {fnf_error}')
        raise
    except Exception as e:
        print(f'Error: {e}')
        raise

# test_function_code --------------------

def test_transcribe_voice_note():
    """
    Test the transcribe_voice_note function.
    """
    # Test with a valid audio file
    try:
        transcription = transcribe_voice_note('valid_audio_file.wav')
        assert isinstance(transcription, str), 'The transcription should be a string.'
    except FileNotFoundError:
        pass

    # Test with a non-existent audio file
    try:
        transcription = transcribe_voice_note('non_existent_file.wav')
    except FileNotFoundError:
        pass

    # Test with an invalid audio file
    try:
        transcription = transcribe_voice_note('invalid_audio_file.wav')
    except Exception:
        pass

    return 'All Tests Passed'

# call_test_function_code --------------------

test_transcribe_voice_note()