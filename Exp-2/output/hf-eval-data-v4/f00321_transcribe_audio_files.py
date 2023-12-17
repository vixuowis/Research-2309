# requirements_file --------------------

!pip install -U transformers datasets

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio_files(audio_data):
    """
    Transcribe a list of audio recordings into text using Whisper ASR model.

    Parameters:
    audio_data (list): A list of dictionaries, where each dictionary contains 'audio' and 'sampling_rate'.

    Returns:
    list: A list of transcribed texts.
    """
    # Load the pre-trained Whisper model
    processor = WhisperProcessor.from_pretrained('openai/whisper-tiny.en')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny.en')

    # Transcribe all audio data
    transcriptions = []
    for audio in audio_data:
        input_features = processor(audio['audio'], sampling_rate=audio['sampling_rate'], return_tensors='pt').input_features
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcriptions.append(transcription)
    return transcriptions

# test_function_code --------------------

def test_transcribe_audio_files():
    print("Testing started.")
    dataset = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample_data = dataset[:3]  # Get 3 samples from the dataset
    audio_data = [{'audio': sample['array'], 'sampling_rate': sample['sampling_rate']} for sample in sample_data]

    transcriptions = transcribe_audio_files(audio_data)
    assert len(transcriptions) == 3, f"Test failed: Expected 3 transcriptions, got {len(transcriptions)}"
    for idx, transcription in enumerate(transcriptions):
        print(f"Testing case [{idx+1}/3] started.")
        assert isinstance(transcription, str), f"Test case [{idx+1}/3] failed: The transcription is not a string."
    print("Testing finished.")

test_transcribe_audio_files()