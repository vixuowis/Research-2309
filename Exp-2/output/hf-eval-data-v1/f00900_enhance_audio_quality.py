from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

def enhance_audio_quality(path_to_low_quality_audio):
    """
    Enhances the audio quality of a low-quality audio file using a pre-trained model from SpeechBrain.

    Args:
        path_to_low_quality_audio (str): Path to the low-quality audio file.

    Returns:
        str: Path to the enhanced audio file.
    """
    model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement', savedir='pretrained_models/sepformer-wham16k-enhancement')
    est_sources = model.separate_file(path=path_to_low_quality_audio)
    enhanced_audio_path = 'enhanced_audio.wav'
    torchaudio.save(enhanced_audio_path, est_sources[:, :, 0].detach().cpu(), 16000)
    return enhanced_audio_path