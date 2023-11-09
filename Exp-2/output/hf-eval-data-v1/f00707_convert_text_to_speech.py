import torch
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech

def convert_text_to_speech(text):
    '''
    This function converts a given text into synthesized speech using a pretrained Text-to-Speech model from ESPnet.
    
    Args:
    text (str): The text message to be converted into speech.
    
    Returns:
    wav (Tensor): The synthesized speech in the form of a waveform.
    '''
    # Download the model
    d = ModelDownloader()
    
    # Load the pretrained Text-to-Speech model
    text2speech = Text2Speech.from_pretrained('espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan')
    
    # Convert the text into speech
    wav, _, _ = text2speech(text)
    
    return wav