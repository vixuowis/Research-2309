from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf


def generate_audio_announcement(text):
    '''
    This function generates an audio announcement from a given text using the SpeechT5 model from Hugging Face Transformers.
    Args:
    text (str): The text to be converted to speech.
    Returns:
    None. The function writes the output audio to a .wav file.
    '''
    # Load the SpeechT5 processor, the Text-to-Speech model, and the Hifigan vocoder
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
    model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')
    
    # Prepare the input for the model
    inputs = processor(text=text, return_tensors='pt')
    
    # Load speaker embeddings to enhance the quality of the synthesized speech
    embeddings_dataset = load_dataset('Matthijs/cmu-arctic-xvectors', split='validation')
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]['xvector']).unsqueeze(0)
    
    # Generate the speech
    speech = model.generate_speech(inputs['input_ids'], speaker_embeddings, vocoder=vocoder)
    
    # Write the speech to a .wav file
    sf.write('speech.wav', speech.numpy(), samplerate=16000)