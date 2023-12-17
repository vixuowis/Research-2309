# requirements_file --------------------

!pip install -U speechbrain, torchaudio

# function_import --------------------

from speechbrain.pretrained import EncoderClassifier
import torchaudio

# function_code --------------------

def identify_language(audio_url):
    """
    Recognize the language spoken in the online audio file.

    Parameters:
    audio_url (str): URL of the online audio file to be analyzed.

    Returns:
    str: Predicted language label.
    """
    # Load the pre-trained language recognition model
    language_id = EncoderClassifier.from_hparams(source='speechbrain/lang-id-voxlingua107-ecapa', savedir='/tmp')
    
    # Load the audio file from the URL
    signal = language_id.load_audio(audio_url)
    
    # Perform language identification on the waveform
    prediction = language_id.classify_batch(signal)
    
    # Extract the predicted language label
    predicted_language = prediction[0][0][1]
    
    return predicted_language

# test_function_code --------------------

