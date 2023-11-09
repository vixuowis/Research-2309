from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio


def separate_audio_sources(input_audio_file):
    '''
    This function separates the background music and vocal from an audio file using the SepFormer model from SpeechBrain.
    
    Args:
    input_audio_file (str): Path to the input audio file.
    
    Returns:
    None. The function saves the separated sources to new audio files.
    '''
    # Load the pre-trained SepFormer model
    model = separator.from_hparams(source='speechbrain/sepformer-wsj02mix', savedir='pretrained_models/sepformer-wsj02mix')
    
    # Separate the sources in the given audio file
    est_sources = model.separate_file(path=input_audio_file)
    
    # Save the separated sources to new audio files
    torchaudio.save('source1hat.wav', est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save('source2hat.wav', est_sources[:, :, 1].detach().cpu(), 8000)