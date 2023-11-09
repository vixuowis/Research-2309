# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration

# function_code --------------------

def transcribe_audio(audio_data, audio_sampling_rate):
    """
    Transcribe audio data using the Whisper ASR model.

    Args:
        audio_data (np.array): The audio data to be transcribed.
        audio_sampling_rate (int): The sampling rate of the audio data.

    Returns:
        str: The transcription of the audio data.
    """
    # Initialize the WhisperProcessor and the WhisperForConditionalGeneration model
    processor = WhisperProcessor.from_pretrained('openai/whisper-large-v2')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v2')

    # Process the audio data with the processor to generate input features
    input_features = processor(audio_data, sampling_rate=audio_sampling_rate, return_tensors='pt').input_features

    # Use the Whisper ASR model to generate the predicted_ids from the input_features
    predicted_ids = model.generate(input_features)

    # Decode the predicted_ids to obtain the transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription

# test_function_code --------------------

def test_transcribe_audio():
    """
    Test the transcribe_audio function.
    """
    from datasets import load_dataset

    # Load a sample from the LibriSpeech dataset
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample = ds[0]['audio']

    # Transcribe the audio sample
    transcription = transcribe_audio(sample['array'], sample['sampling_rate'])

    # Assert that the transcription is not empty
    assert transcription, 'The transcription is empty.'

# call_test_function_code --------------------

test_transcribe_audio()