from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# Function to analyze the emotion of the speaker in an audio recording
# Uses the Hugging Face Transformers library and the 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition' model
# @param audio_path: The path to the audio file to analyze
# @return: The predicted emotion of the speaker

def analyze_speaker_emotion(audio_path):
    # Load the pre-trained emotion recognition model
    model = Wav2Vec2ForCTC.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')
    # Load the tokenizer
    tokenizer = Wav2Vec2Processor.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')
    # Process the audio file and convert it into the required format for the model
    input_data = tokenizer(audio_path, return_tensors="pt")
    input_values = input_data.input_values.to("cuda")
    # Pass the processed audio file to the model and analyze the speaker's emotion
    predictions = model(input_values)
    predicted_ids = torch.argmax(predictions.logits, dim=-1)
    # Decode the predicted ids to get the predicted emotions
    predicted_emotions = tokenizer.batch_decode(predicted_ids)
    return predicted_emotions