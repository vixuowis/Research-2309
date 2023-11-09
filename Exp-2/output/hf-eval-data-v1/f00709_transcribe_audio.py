from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# Function to transcribe audio using the Whisper model
# @param audio_samples: A list of audio samples to transcribe
# @return: A list of transcriptions

def transcribe_audio(audio_samples):
    # Load the Whisper processor and model
    processor = WhisperProcessor.from_pretrained('openai/whisper-small')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')

    # Initialize a list to store the transcriptions
    transcriptions = []

    # Iterate through the audio samples
    for audio_sample in audio_samples:
        # Process the audio sample to create the input features
        input_features = processor(audio_sample['array'], sampling_rate=audio_sample['sampling_rate'], return_tensors='pt').input_features
        # Use the model to generate the predicted token IDs
        predicted_ids = model.generate(input_features)
        # Decode the token IDs to get the transcription
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        # Add the transcription to the list
        transcriptions.append(transcription)

    # Return the list of transcriptions
    return transcriptions