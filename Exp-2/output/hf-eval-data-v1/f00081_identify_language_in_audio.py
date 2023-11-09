from transformers import AutoModelForSpeechClassification, Wav2Vec2Processor
import torch

# Function to identify language in an audio file
# Uses the Hugging Face Transformers library and a pre-trained model for speech classification
# The model has been fine-tuned for identifying languages in audio data

def identify_language_in_audio(audio_file):
    # Load the pre-trained model and processor
    model = AutoModelForSpeechClassification.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')
    processor = Wav2Vec2Processor.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')

    # Process the audio file
    input_values = processor(audio_file, return_tensors='pt').input_values

    # Predict the language
    with torch.no_grad():
        logits = model(input_values).logits

    # Get the predicted language
    predicted_language = torch.argmax(logits, dim=-1)

    return predicted_language.item()