# requirements_file --------------------

!pip install -U diffusers transformers accelerate controlnet_aux Pillow torch torchvision requests

# function_import --------------------

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import MLSDdetector
from diffusers.utils import load_image

# function_code --------------------

def process_image_with_controlnet(image_path: str, output_path: str) -> None:
    """
    Processes an image by detecting straight lines and controlling the diffusion process using Hugging Face's ControlNet.

    Args:
        image_path: A string path to the input image file.
        output_path: A string path where the processed image will be saved.

    Raises:
        FileNotFoundError: If the input image file does not exist.
    """
    mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-mlsd', torch_dtype=torch.float16)
    image = load_image(image_path)
    image = mlsd(image)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',controlnet=controlnet,safety_checker=None,torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    processed_image = pipe(image, num_inference_steps=20).images[0]
    processed_image.save(output_path)

# test_function_code --------------------

def test_process_image_with_controlnet():
    print("Testing started.")
    test_image_url = 'https://huggingface.co/lllyasviel/sd-controlnet-mlsd/resolve/main/images/room.png'
    test_image_path = 'test_input_image.png'
    download_file(test_image_url, test_image_path)
    output_image_path = 'test_output_image.png'
    print("Testing case [1/1] started.")
    try:
        process_image_with_controlnet(test_image_path, output_image_path)
        assert os.path.exists(output_image_path), "Output image file was not created."
    except Exception as e:
        assert False, f"Test case [1/1] failed: {e}"
    print("Testing finished.")
def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

# call_test_function_line --------------------

test_process_image_with_controlnet()