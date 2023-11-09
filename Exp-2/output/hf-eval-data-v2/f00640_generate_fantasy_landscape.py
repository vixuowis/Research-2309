# function_import --------------------

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

# function_code --------------------

def generate_fantasy_landscape(prompt: str, model_id: str = 'darkstorm2150/Protogen_v5.8_Official_Release', torch_dtype: str = 'torch.float16', num_inference_steps: int = 25) -> None:
    """
    Generate an image of a fantasy landscape based on the description provided.

    Args:
        prompt (str): The description of the fantasy landscape.
        model_id (str, optional): The model id to be used for image generation. Defaults to 'darkstorm2150/Protogen_v5.8_Official_Release'.
        torch_dtype (str, optional): The torch data type to be used. Defaults to 'torch.float16'.
        num_inference_steps (int, optional): The number of inference steps to be used for image generation. Defaults to 25.

    Returns:
        None. The function saves the generated image to a file.
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to('cuda')
    image_result = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
    image_result.save('./result.jpg')

# test_function_code --------------------

def test_generate_fantasy_landscape():
    """
    Test the function generate_fantasy_landscape.

    The function is tested with a sample prompt. The test passes if the function runs without throwing any exceptions.
    """
    prompt = 'a peaceful scene in a lush green forest with a crystal-clear river flowing through it, under a blue sky with fluffy white clouds'
    try:
        generate_fantasy_landscape(prompt)
        print('Test passed.')
    except Exception as e:
        print(f'Test failed. {str(e)}')

# call_test_function_code --------------------

test_generate_fantasy_landscape()