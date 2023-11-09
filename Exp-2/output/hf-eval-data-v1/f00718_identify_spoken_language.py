from transformers import AutoModelForSpeechClassification, Wav2Vec2Processor
import soundfile as sf

# Function to identify the language spoken in an audio file
# Uses the Hugging Face Transformers library and a pretrained model for speech classification
# The model has been fine-tuned on the FLEURS subset of the google/xtreme_s dataset
# Returns the ID of the predicted language

def identify_spoken_language(audio_file_path):
    # Load the pretrained language identification model
    model = AutoModelForSpeechClassification.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')
    # Load the corresponding Wav2Vec2Processor for preprocessing the audio data
    processor = Wav2Vec2Processor.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')
    # Read the audio file
    audio, sample_rate = sf.read(audio_file_path)
    # Preprocess the audio data
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    # Pass the preprocessed audio data to the model
    logits = model(**inputs).logits
    # Get the ID of the predicted language
    predicted_language_id = logits.argmax(-1).item()
    return predicted_language_id