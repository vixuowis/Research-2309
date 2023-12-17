# requirements_file --------------------

!pip install -U transformers datasets

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_speech_to_text(audio_sample):
    """
    Transcribes speech to text using the pre-trained Whisper ASR model.

    Args:
        audio_sample (dict): An audio sample with 'array' and 'sampling_rate'.

    Returns:
        str: The transcribed text.

    Raises:
        ValueError: If 'audio_sample' is not a dict or does not have necessary keys.
    """
    if not isinstance(audio_sample, dict):
        raise ValueError("The 'audio_sample' should be a dictionary.")
    if 'array' not in audio_sample or 'sampling_rate' not in audio_sample:
        raise ValueError("'audio_sample' must contain 'array' and 'sampling_rate'.")

    processor = WhisperProcessor.from_pretrained('openai/whisper-base')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base')

    input_features = processor(audio_sample['array'], 
                               sampling_rate=audio_sample['sampling_rate'], 
                               return_tensors='pt').input_features

    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription

# test_function_code --------------------

def test_transcribe_speech_to_text():
    print("Testing started.")
    dataset = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')

    # Expected output is known for dummy dataset
    expected_output = "A MAN SAID TO THE UNIVERSE SIR I EXIST"

    # Testing case 1
    print("Testing case [1/1] started.")
    sample = dataset[0]['audio']
    transcription = transcribe_speech_to_text(sample)
    assert transcription == expected_output, f"Test case [1/1] failed: expected {expected_output}, got {transcription}"
    print("Testing finished.")

# call_test_function_line --------------------

test_transcribe_speech_to_text()