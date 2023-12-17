# requirements_file --------------------

!pip install -U transformers datasets

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration

# function_code --------------------

def transcribe_audio(audio_file_path):
    """
    Transcribe the given audio file using the Whisper ASR model.

    :param audio_file_path: str
        The path to the audio file that needs to be transcribed.
    :return: str
        The transcribed text.
    """
    processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')

    # Load and process the audio file
    sample = {'array': audio_file_path, 'sampling_rate': 16000}
    input_features = processor(sample['array'], sampling_rate=sample['sampling_rate'], return_tensors='pt').input_features

    # Generate transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription

# test_function_code --------------------

def test_transcribe_audio():
    print("Testing transcribe_audio function.")
    sample_audio_file = 'test_audio_sample.wav'  # This should be replaced with a real audio file for testing

    print("Testing with sample audio file.")
    transcription = transcribe_audio(sample_audio_file)
    assert len(transcription) > 0, f"Transcription failed, no text output detected."
    print("Test passed, transcription was successful.")

# Run the test function
test_transcribe_audio()