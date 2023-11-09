from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

# Function to convert audio to text using Wav2Vec2ForCTC model
def convert_audio_to_text(audio_file):
    '''
    This function converts an audio file to text using the Wav2Vec2ForCTC model from the Transformers library.
    The model has been pre-trained on the 'facebook/wav2vec2-base-960h' dataset.
    
    Parameters:
    audio_file (str): Path to the audio file
    
    Returns:
    str: The transcribed text from the audio file
    '''
    # Load the pre-trained model and processor
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')

    # Pre-process the audio file
    input_values = processor(audio_file, return_tensors='pt', padding='longest').input_values

    # Get logits from the model
    logits = model(input_values).logits

    # Predict the transcriptions
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode transcriptions into text
    transcription = processor.batch_decode(predicted_ids)

    return transcription