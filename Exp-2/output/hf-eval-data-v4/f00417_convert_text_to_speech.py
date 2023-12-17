# requirements_file --------------------

!pip install -U fairseq IPython

# function_import --------------------

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

# function_code --------------------

def convert_text_to_speech(message):
    """
    Convert text to speech using a pre-trained Fairseq FastSpeech 2 model.
    :param message: The text message to convert to speech.
    :return: IPython.display Audio object with the generated audio waveform
    """
    # Load the pre-trained FastSpeech 2 model from Hugging Face Hub
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub('facebook/fastspeech2-en-ljspeech', arg_overrides={'vocoder': 'hifigan', 'fp16': False})
    model = models[0]
    # Update the configuration with the corresponding data config
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    # Build a generator for the text-to-speech conversion
    generator = task.build_generator(models, cfg)
    # Convert the sensitive warning message into a model input
    sample = TTSHubInterface.get_model_input(task, message)
    # Generate the audio waveform
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    # Create an audio object for playback
    audio_output = ipd.Audio(wav, rate=rate)
    return audio_output

# test_function_code --------------------

def test_convert_text_to_speech():
    print("Testing convert_text_to_speech function.")
    test_message = "This is a test run of the Text-to-Speech system."

    # Test the text-to-speech conversion function
    print("Testing case [1/1] started.")
    audio_output = convert_text_to_speech(test_message)

    assert isinstance(audio_output, ipd.Audio), "The function did not return an IPython.display Audio object."

    print("Testing finished.")

test_convert_text_to_speech()