import torchaudio
from speechbrain.pretrained import WaveformEnhancement

def enhance_audio(input_audio_file: str, output_audio_file: str) -> None:
    """
    Enhances the quality of an audio file by reducing background noise.

    Args:
        input_audio_file (str): The path to the input audio file that needs enhancement.
        output_audio_file (str): The path where the enhanced audio file will be saved.

    Returns:
        None
    """
    enhance_model = WaveformEnhancement.from_hparams(
        source="speechbrain/mtl-mimic-voicebank",
        savedir="pretrained_models/mtl-mimic-voicebank",
    )
    enhanced = enhance_model.enhance_file(input_audio_file)
    torchaudio.save(output_audio_file, enhanced.unsqueeze(0).cpu(), 16000)