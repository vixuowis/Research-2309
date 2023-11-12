# function_import --------------------

from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

# function_code --------------------

def load_text_to_image_model(model_name: str, vae_model_name: str):
    """
    Load a text-to-image conversion model.

    Args:
        model_name (str): The name of the pre-trained model to load.
        vae_model_name (str): The name of the VAE model to load.

    Returns:
        StableDiffusionPipeline: The loaded model.

    Raises:
        ModuleNotFoundError: If the required libraries are not installed.
    """
    vae = AutoencoderKL.from_pretrained(vae_model_name)
    model = StableDiffusionPipeline.from_pretrained(model_name, vae=vae)
    return model

# test_function_code --------------------

def test_load_text_to_image_model():
    """
    Test the load_text_to_image_model function.
    """
    try:
        model = load_text_to_image_model('CompVis/stable-diffusion-v1-4', 'stabilityai/sd-vae-ft-ema')
        assert isinstance(model, StableDiffusionPipeline)
        print('Test passed.')
    except Exception as e:
        print('Test failed.\n', e)

# call_test_function_code --------------------

test_load_text_to_image_model()