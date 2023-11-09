from transformers import Text2Speech
import torch


def text_to_speech(text):
    """
    This function converts text into speech using a pretrained model from ESPnet.
    The model used is 'kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan'.
    
    Args:
    text (str): The text to be converted into speech.
    
    Returns:
    Tensor: The synthesized speech output.
    """
    # Load the pretrained model
    model = Text2Speech.from_pretrained('espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan')
    
    # Convert the text into speech
    speech_output = model(text)
    
    return speech_output