from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio


def enhance_speech(input_audio_path: str, output_audio_path: str = 'enhanced_audio.wav'):
    """
    This function enhances the speech quality in an audio file using the SepFormer model from SpeechBrain.
    
    Args:
        input_audio_path (str): Path to the input audio file.
        output_audio_path (str, optional): Path to save the enhanced audio file. Defaults to 'enhanced_audio.wav'.
    
    Returns:
        None
    """
    # Load the pre-trained SepFormer model
    model = separator.from_hparams(source='speechbrain/sepformer-wham-enhancement', savedir='pretrained_models/sepformer-wham-enhancement')
    
    # Enhance the speech in the audio file
    est_sources = model.separate_file(path=input_audio_path)
    
    # Save the enhanced audio file
    torchaudio.save(output_audio_path, est_sources[:, :, 0].detach().cpu(), 8000)