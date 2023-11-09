import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement


def enhance_audio(input_file: str, output_file: str):
    """
    This function enhances the audio quality of a given input file using a pre-trained MetricGAN model
    from the Hugging Face Model Hub and saves the enhanced audio to the specified output file.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the enhanced audio file.
    """
    # Load the pre-trained MetricGAN model
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source='speechbrain/metricgan-plus-voicebank',
        savedir='pretrained_models/metricgan-plus-voicebank',
    )
    # Load the noisy audio file
    noisy = enhance_model.load_audio(input_file).unsqueeze(0)
    # Enhance the audio
    enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))
    # Save the enhanced audio
    torchaudio.save(output_file, enhanced.cpu(), 16000)