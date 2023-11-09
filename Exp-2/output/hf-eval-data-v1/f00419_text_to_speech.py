from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

def text_to_speech(text):
    '''
    This function converts the given text into speech using the FastSpeech 2 model.
    Args:
    text : str
        The text to be converted into speech.
    Returns:
    Audio
        The audio output of the converted text.
    '''
    # Load the FastSpeech 2 model
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        'facebook/fastspeech2-en-200_speaker-cv4',
        arg_overrides={'vocoder': 'hifigan', 'fp16': False}
    )
    model = models[0]
    # Update the model's configuration
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    # Build the generator function
    generator = task.build_generator(model, cfg)
    # Get the model input
    sample = TTSHubInterface.get_model_input(task, text)
    # Get the prediction
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    # Return the audio output
    return ipd.Audio(wav, rate=rate)