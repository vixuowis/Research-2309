# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio(audio_file_path):
    """
    Transcribe an audio file using the Whisper ASR model.

    Args:
        audio_file_path (str): The path to the audio file to transcribe.

    Returns:
        str: The transcription of the audio file.
    """
    # Load the processor and model
    processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')

    # Load the audio file
    sample = {'array': audio_file_path, 'sampling_rate': 16000}

    # Preprocess the audio file
    input_features = processor(sample['array'], sampling_rate=sample['sampling_rate'], return_tensors='pt').input_features

    # Generate the transcription
    predicted_ids = model.generate(input_features)

    # Decode the transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription

# test_function_code --------------------

def test_transcribe_audio():
    """
    Test the transcribe_audio function.
    """
    # Load a sample audio file from the LibriSpeech dataset
    ds = load_dataset('librispeech_asr', 'clean', split='validation')
    sample = ds[0]['audio']

    # Transcribe the audio file
    transcription = transcribe_audio(sample)

    # Check that the transcription is not empty
    assert transcription != ''

# call_test_function_code --------------------

test_transcribe_audio()