from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# Function to enhance audio by reducing noise
# Uses the pre-trained Sepformer model from SpeechBrain
# @param audio_path: Path to the input audio file
# @return: Path to the enhanced audio file

def enhance_audio(audio_path):
    # Load the pre-trained separator model
    model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement', savedir='pretrained_models/sepformer-wham16k-enhancement')
    # Separate speech from the noise in the target audio file
    est_sources = model.separate_file(path=audio_path)
    # Save the enhanced audio file to disk
    enhanced_audio_path = 'enhanced_' + audio_path
    torchaudio.save(enhanced_audio_path, est_sources[:, :, 0].detach().cpu(), 16000)
    return enhanced_audio_path