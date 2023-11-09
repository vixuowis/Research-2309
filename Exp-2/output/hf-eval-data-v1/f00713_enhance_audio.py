from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio


def enhance_audio(input_audio_file: str, output_audio_file: str = 'enhanced_audio_file.wav') -> None:
    """
    This function enhances the audio by removing noise using a pre-trained model from SpeechBrain.
    
    Args:
        input_audio_file (str): Path to the input audio file.
        output_audio_file (str, optional): Path to save the enhanced audio file. Defaults to 'enhanced_audio_file.wav'.
    
    Returns:
        None
    """
    # Load the pre-trained model
    model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement', savedir='pretrained_models/sepformer-wham16k-enhancement')
    
    # Denoise the input audio file
    est_sources = model.separate_file(path=input_audio_file)
    
    # Save the enhanced audio file
    torchaudio.save(output_audio_file, est_sources[:, :, 0].detach().cpu(), 16000)