from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset


def transcribe_audio(audio_sample):
    """
    This function transcribes an audio sample using the Whisper ASR model.
    
    Parameters:
    audio_sample (dict): A dictionary containing the audio array and the sampling rate.
    
    Returns:
    str: The transcribed text.
    """
    # Load the Whisper processor and model
    processor = WhisperProcessor.from_pretrained('openai/whisper-base')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base')
    
    # Preprocess the audio sample
    input_features = processor(audio_sample['array'], sampling_rate=audio_sample['sampling_rate'], return_tensors='pt').input_features
    
    # Generate the predicted IDs
    predicted_ids = model.generate(input_features)
    
    # Decode the predicted IDs to get the transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription