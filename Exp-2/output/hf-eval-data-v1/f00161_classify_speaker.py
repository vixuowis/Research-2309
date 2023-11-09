import torchaudio
from speechbrain.pretrained import EncoderClassifier

def classify_speaker(audio_file_path):
    '''
    This function takes an audio file path as input and returns the speaker embeddings.
    It uses the pre-trained speaker verification model 'speechbrain/spkrec-xvect-voxceleb'.
    
    Parameters:
    audio_file_path (str): The path to the audio file.
    
    Returns:
    embeddings (tensor): The speaker embeddings for the audio file.
    '''
    # Instantiate the pre-trained speaker verification model
    classifier = EncoderClassifier.from_hparams(source='speechbrain/spkrec-xvect-voxceleb', savedir='pretrained_models/spkrec-xvect-voxceleb')
    # Load the audio file
    signal, fs = torchaudio.load(audio_file_path)
    # Generate speaker embeddings
    embeddings = classifier.encode_batch(signal)
    return embeddings