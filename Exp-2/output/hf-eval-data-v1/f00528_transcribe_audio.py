from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import torch

# Function to transcribe audio to text
# This function uses the Hugging Face Transformers library to transcribe audio files into text.
# The function uses the Wav2Vec2ForCTC model, which is pretrained on the 'jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli' dataset.
# This model is particularly suitable for creating transcriptions with accurate punctuation.
def transcribe_audio(audio_file):
    # Load the pretrained model
    model = Wav2Vec2ForCTC.from_pretrained('jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli')
    # Load the processor
    processor = Wav2Vec2Processor.from_pretrained('jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli')
    # Read the audio file
    speech, _ = sf.read(audio_file)
    # Process the audio file
    input_values = processor(speech, return_tensors='pt').input_values
    # Make the prediction
    logits = model(input_values).logits
    # Decode the prediction
    predicted_ids = torch.argmax(logits, dim=-1)
    # Convert the prediction into text
    transcription = processor.decode(predicted_ids[0])
    # Return the transcription
    return transcription