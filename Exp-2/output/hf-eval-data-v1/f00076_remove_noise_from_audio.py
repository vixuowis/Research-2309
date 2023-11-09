import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement


def remove_noise_from_audio(input_audio_path: str, output_audio_path: str = 'enhanced.wav'):
    """
    This function removes noise from an audio file using a pre-trained model from SpeechBrain.
    
    Args:
        input_audio_path (str): Path to the input noisy audio file.
        output_audio_path (str, optional): Path to save the enhanced audio file. Defaults to 'enhanced.wav'.
    
    Returns:
        None
    """
    # Load the pre-trained speech enhancement model
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source='speechbrain/metricgan-plus-voicebank',
        savedir='pretrained_models/metricgan-plus-voicebank',
    )
    
    # Load the noisy audio file
    noisy = enhance_model.load_audio(input_audio_path).unsqueeze(0)
    
    # Enhance the audio
    enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))
    
    # Save the enhanced audio to a file
    torchaudio.save(output_audio_path, enhanced.cpu(), 16000)