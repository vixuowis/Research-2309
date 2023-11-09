from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
import numpy as np
import soundfile as sf
import torch

def convert_voice(input_audio_path, speaker_embedding_path, output_audio_path):
    '''
    This function converts the voice in an audio file to a different voice without changing the content.
    It uses the SpeechT5 model from Hugging Face Transformers.
    
    Parameters:
    input_audio_path (str): The path to the input audio file.
    speaker_embedding_path (str): The path to the speaker embedding file.
    output_audio_path (str): The path to save the output audio file.
    
    Returns:
    None
    '''
    # Load the example speech from file and retrieve the sampling rate
    example_speech, sampling_rate = sf.read(input_audio_path)
    
    # Create instances of the SpeechT5Processor, SpeechT5ForSpeechToSpeech, and SpeechT5HifiGan
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_vc')
    model = SpeechT5ForSpeechToSpeech.from_pretrained('microsoft/speecht5_vc')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')
    
    # Preprocess the input audio
    inputs = processor(audio=example_speech, sampling_rate=sampling_rate, return_tensors='pt')
    
    # Load speaker embeddings for the desired target voice
    speaker_embeddings = np.load(speaker_embedding_path)
    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
    
    # Generate the converted speech
    speech = model.generate_speech(inputs['input_values'], speaker_embeddings, vocoder=vocoder)
    
    # Save the resulting speech as a .wav file
    sf.write(output_audio_path, speech.numpy(), samplerate=16000)