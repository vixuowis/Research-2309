from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import soundfile as sf

# Load the pre-trained model and processor
model = Wav2Vec2ForSequenceClassification.from_pretrained('hackathon-pln-es/wav2vec2-base-finetuned-sentiment-classification-MESD')
processor = Wav2Vec2Processor.from_pretrained('hackathon-pln-es/wav2vec2-base-finetuned-sentiment-classification-MESD')

def classify_sentiment(audio_file):
    '''
    This function takes an audio file as input and returns the sentiment of the speech in the audio.
    It uses a pre-trained model from Hugging Face Transformers for sentiment classification.
    '''
    # Read the audio file
    speech, _ = sf.read(audio_file)
    # Preprocess the audio data
    inputs = processor(speech, return_tensors='pt', padding=True)
    # Feed the preprocessed data into the model
    logits = model(**inputs).logits
    # Get the predicted sentiment
    pred_ids = logits.argmax(dim=-1).item()
    # Convert the prediction to a sentiment label
    label = processor.tokenizer.convert_ids_to_tokens([pred_ids])[0]
    return label