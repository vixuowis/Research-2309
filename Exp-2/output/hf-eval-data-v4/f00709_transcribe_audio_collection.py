# requirements_file --------------------

!pip install -U transformers datasets

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio_collection(audio_samples):
    """
    Transcribe a collection of audio samples into text using Whisper.

    Parameters:
        audio_samples (list): A list of audio sample dictionaries with 'array' and 'sampling_rate' keys.

    Returns:
        List[str]: A list containing the transcribed texts.
    """
    processor = WhisperProcessor.from_pretrained('openai/whisper-small')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')

    transcriptions = []
    for audio_sample in audio_samples:
        input_features = processor(audio_sample['array'], sampling_rate=audio_sample['sampling_rate'], return_tensors='pt').input_features
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        transcriptions.append(transcription[0])

    return transcriptions

# test_function_code --------------------

def test_transcribe_audio_collection():
    print("Testing started.")
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample_data = dataset[:3]  # Take a few samples for testing

    # Execute the transcription function
    transcribed_texts = transcribe_audio_collection(sample_data['audio'])

    # Check whether transcriptions are strings and not empty
    for i, text in enumerate(transcribed_texts):
        assert isinstance(text, str) and len(text) > 0, f"Test case [{i+1}] failed: Transcription output is not a valid non-empty string."

    print("Testing finished.")

# Run the test function
test_transcribe_audio_collection()