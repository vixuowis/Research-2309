from transformers import Asteroid
import torch
import soundfile as sf


def remove_overlapping_speakers(input_file_path, output_file_path):
    """
    This function uses the 'ConvTasNet_Libri2Mix_sepclean_16k' model from Hugging Face Transformers to remove overlapping speakers from an audio recording.
    
    Parameters:
    input_file_path (str): The path to the input mixed audio file.
    output_file_path (str): The path to the output separated audio file.
    
    Returns:
    None
    """
    # Load the model
    model = Asteroid('JorisCos/ConvTasNet_Libri2Mix_sepclean_16k')
    
    # Read the mixed audio file
    mixed_audio, sample_rate = sf.read(input_file_path)
    
    # Convert the audio to a tensor
    mixed_audio_tensor = torch.tensor(mixed_audio)
    
    # Use the model to separate the speakers
    separated_audio_tensor = model(mixed_audio_tensor)
    
    # Convert the tensor back to numpy
    separated_audio = separated_audio_tensor.numpy()
    
    # Write the separated audio to a file
    sf.write(output_file_path, separated_audio, sample_rate)