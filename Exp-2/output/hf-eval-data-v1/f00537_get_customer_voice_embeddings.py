import torchaudio
from speechbrain.pretrained import EncoderClassifier

def get_customer_voice_embeddings(audio_file):
    '''
    This function takes an audio file as input and returns the voice embeddings of the customer.
    It uses a pre-trained speaker recognition model from SpeechBrain.
    
    Args:
    audio_file (str): Path to the audio file.
    
    Returns:
    embeddings (tensor): Voice embeddings of the customer.
    '''
    # Load the pre-trained speaker recognition model
    classifier = EncoderClassifier.from_hparams(source='speechbrain/spkrec-xvect-voxceleb', savedir='pretrained_models/spkrec-xvect-voxceleb')
    
    # Load the audio file
    signal, fs = torchaudio.load(audio_file)
    
    # Generate voice embeddings for the audio file
    embeddings = classifier.encode_batch(signal)
    
    return embeddings