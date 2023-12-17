# requirements_file --------------------

!pip install -U transformers datasets

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio(audio_path):
    """
    Transcribe the given audio file using the Whisper model.

    :param audio_path: Path to the audio file to be transcribed.
    :return: The transcribed text.
    """
    processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')

    # Load audio
    audio = get_audio_sample(audio_path)

    # Prepare the audio input
    input_features = processor(audio['array'], sampling_rate=audio['sampling_rate'], return_tensors='pt').input_features

    # Generate transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription

# test_function_code --------------------

def test_transcribe_audio():
    print('Testing transcribe_audio function.')

    # Prepare a path to a demo audio file
    audio_path = 'path/to/demo_audio.flac'

    # Call the transcription function
    transcription = transcribe_audio(audio_path)

    # Check if the transcription is not empty
    assert transcription, 'Transcription is empty.'

    print('Test passed!')