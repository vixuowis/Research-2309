from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch

# Function to classify sentiment in Spanish audio
# This function uses the Wav2Vec2ForSequenceClassification model from Hugging Face's transformers library
# The model has been fine-tuned for sentiment classification in Spanish
# The function takes as input the path to an audio file and returns the predicted sentiment

def classify_sentiment(audio_file):
    # Load the pre-trained model
    model = Wav2Vec2ForSequenceClassification.from_pretrained('hackathon-pln-es/wav2vec2-base-finetuned-sentiment-classification-MESD')
    # Load the pre-trained processor
    processor = Wav2Vec2Processor.from_pretrained('hackathon-pln-es/wav2vec2-base-finetuned-sentiment-classification-MESD')
    # Load the audio file
    audio_input = processor(audio_file, return_tensors='pt')
    # Make a prediction
    outputs = model(audio_input.input_values)
    # Get the predicted class
    predicted_class = torch.argmax(outputs.logits, dim=-1)
    # Return the predicted sentiment
    return predicted_class