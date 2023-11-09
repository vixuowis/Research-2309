from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

def enhance_audio(input_audio_file, output_audio_file):
    '''
    This function enhances the audio of noisy recordings using the Sepformer model trained on the WHAMR! dataset.
    Args:
    input_audio_file (str): Path to the input audio file.
    output_audio_file (str): Path to save the enhanced audio file.
    Returns:
    None
    '''
    # Load the trained Sepformer model
    model = separator.from_hparams(source='speechbrain/sepformer-whamr-enhancement', savedir='pretrained_models/sepformer-whamr-enhancement')
    # Enhance the speech by inputting a path to an audio file
    est_sources = model.separate_file(path=input_audio_file)
    # Save the enhanced audio to a file
    torchaudio.save(output_audio_file, est_sources[:, :, 0].detach().cpu(), 8000)