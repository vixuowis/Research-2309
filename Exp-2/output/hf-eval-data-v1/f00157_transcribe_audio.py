from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch


def transcribe_audio(audio_sample):
    """
    This function transcribes an audio sample using the pre-trained model 'openai/whisper-tiny.en'.
    
    Parameters:
    audio_sample (dict): A dictionary containing the audio data and the sampling rate.
    
    Returns:
    str: The transcribed text.
    """
    # Load the pre-trained model and the processor
    processor = WhisperProcessor.from_pretrained('openai/whisper-tiny.en')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny.en')
    
    # Process the audio sample
    input_features = processor(audio_sample['array'], sampling_rate=audio_sample['sampling_rate'], return_tensors='pt').input_features
    
    # Generate the predicted token ids
    predicted_ids = model.generate(input_features)
    
    # Decode the predicted token ids into textual transcriptions
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription[0]