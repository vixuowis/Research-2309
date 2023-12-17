# requirements_file --------------------

!pip install -U fairseq IPython

# function_import --------------------

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

# function_code --------------------

def french_text_to_speech(text):
    """
    Convert French text to speech using a pre-trained model.

    Parameters:
    text (str): The text to convert to speech.

    Returns:
    IPython.display.Audio: An object that allows audio playback in Jupyter notebooks.
    """
    # Load the pre-trained model
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        'facebook/tts_transformer-fr-cv7_css10',
        arg_overrides={'vocoder': 'hifigan', 'fp16': False}
    )
    model = models[0]

    # Update configuration with data configuration
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)

    # Build the generator
    generator = task.build_generator(model, cfg)

    # Generate speech
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)

    # Return the audio object
    return ipd.Audio(wav, rate=rate)

# test_function_code --------------------

def test_french_text_to_speech():
    print("Testing french_text_to_speech function.")

    # Test with a simple phrase
    phrase = "Bonjour, ceci est un test."
    audio_result = french_text_to_speech(phrase)
    assert isinstance(audio_result, ipd.Audio), "The function should return an IPython.display.Audio object."

    # Further tests could include checking the audio duration, quality, etc.
    # These tests require a specific setup, as we are dealing with audio data.

    print("Test passed!")