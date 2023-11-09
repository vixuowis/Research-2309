from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

def enhance_audio(input_file, output_file):
    '''
    This function enhances the quality of an audio file by removing background noise using the SpeechBrain library.
    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the enhanced audio file.
    '''
    # Load the pre-trained model
    model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement', savedir='pretrained_models/sepformer-wham16k-enhancement')
    # Separate the speech from the noise
    est_sources = model.separate_file(path=input_file)
    # Save the enhanced audio to a file
    torchaudio.save(output_file, est_sources[:, :, 0].detach().cpu(), 16000)