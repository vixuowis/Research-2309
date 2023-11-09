import torch
from pyannote.audio import Model, Inference


def measure_noise_levels(audio_file: str, access_token: str) -> dict:
    """
    Measures the noise levels in the environment using a pre-trained model from Hugging Face Transformers.

    Args:
        audio_file (str): The path to the audio file.
        access_token (str): The access token for Hugging Face Transformers.

    Returns:
        dict: A dictionary containing the voice activity detection (VAD), speech-to-noise ratio (SNR), and the C50 room acoustics estimation for each frame in the audio file.
    """
    model = Model.from_pretrained('pyannote/brouhaha', use_auth_token=access_token)
    inference = Inference(model)
    output = inference(audio_file)
    results = {}
    for frame, (vad, snr, c50) in output:
        t = frame.middle
        results[t] = {'vad': vad, 'snr': snr, 'c50': c50}
    return results