# requirements_file --------------------

!pip install -U transformers, datasets, torch, jiwer

# function_import --------------------

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch

# function_code --------------------

def transcribe_podcast_audio(audio_data):
    """
    Transcribes podcast audio data into text using a pre-trained Wav2Vec2 model.

    Args:
        audio_data ([type]): The raw audio data for transcription.

    Returns:
        str: The transcribed text.
    """
    # Initialize the processor and model
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')
    model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')
    
    # Preprocess the audio data
    input_values = processor(audio_data, return_tensors='pt', padding='longest').input_values
    
    # Get logits from the model
    logits = model(input_values).logits
    
    # Decode the logits to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription

# test_function_code --------------------

def test_transcribe_podcast_audio():
    print("Testing transcribe_podcast_audio function.")
    # Load a sample from the dataset
    dataset = load_dataset("librispeech_asr", "clean", split="validation")
    audio_data = dataset[0]['audio']['array']

    # Test case: Transcribe the sample audio
    print("Test case started.")
    transcription = transcribe_podcast_audio(audio_data)
    assert isinstance(transcription, str), f"Transcription should be a string, got: {type(transcription)}"
    assert len(transcription) > 0, "Transcription should not be empty."
    print("Test case passed.")
    print("Testing finished.")

# Run the test function
test_transcribe_podcast_audio()