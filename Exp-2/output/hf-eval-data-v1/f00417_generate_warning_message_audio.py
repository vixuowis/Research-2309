from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

def generate_warning_message_audio(warning_message):
    '''
    This function generates an audio message using the FastSpeech 2 text-to-speech model from fairseq.
    The generated audio can be used in a phonebot to read a sensitive warning message to the users.
    
    Parameters:
    warning_message (str): The warning message to be converted to audio.
    
    Returns:
    IPython.lib.display.Audio: The generated audio message.
    '''
    # Load the pre-trained FastSpeech 2 model
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub('facebook/fastspeech2-en-ljspeech', arg_overrides={'vocoder': 'hifigan', 'fp16': False})
    model = models[0]
    # Update the configuration with the corresponding data config
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    # Build a generator for the text-to-speech conversion
    generator = task.build_generator(model, cfg)
    # Convert the warning message into a model input
    sample = TTSHubInterface.get_model_input(task, warning_message)
    # Generate the audio waveform
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    # Create an audio object that can be played back in the phonebot
    audio_output = ipd.Audio(wav, rate=rate)
    return audio_output