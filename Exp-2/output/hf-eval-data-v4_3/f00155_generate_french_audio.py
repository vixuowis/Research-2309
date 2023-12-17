# requirements_file --------------------

import subprocess

requirements = ["fairseq"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

# function_code --------------------

def generate_french_audio(text: str) -> tuple:
    """
    Generates audio from the given French text using a pre-trained text-to-speech model.

    Args:
        text (str): The French text to be converted to audio.

    Returns:
        tuple: A tuple containing the generated waveform and the sample rate.

    Raises:
        ValueError: If the text is not a string or is empty.
    """

    # Validate input text
    if not isinstance(text, str) or not text:
        raise ValueError('The input text must be a non-empty string.')

    # Load the pre-trained model
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub('facebook/tts_transformer-fr-cv7_css10')
    model = models[0]
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(model, cfg)

    # Generate audio
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    return wav, rate

# test_function_code --------------------

def test_generate_french_audio():
    print("Testing started.")

    # Sample French text
    sample_text = "Bonjour, ceci est un test."

    # Test case 1: Correct input
    print("Testing case [1/1] started.")
    wav, rate = generate_french_audio(sample_text)
    assert isinstance(wav, np.ndarray), "The waveform should be a NumPy array."
    assert isinstance(rate, int), "The sample rate should be an integer."

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_french_audio()