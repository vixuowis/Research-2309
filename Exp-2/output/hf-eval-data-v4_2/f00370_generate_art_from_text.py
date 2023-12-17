# requirements_file --------------------

!pip install -U torch diffusers controlnet_aux

# function_import --------------------

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image
from controlnet_aux import NormalBaeDetector

# function_code --------------------


    def generate_art_from_text(prompt, image_url, controlnet_checkpoint, seed=33):
        """
        Generate a piece of art based on given textual description using the specified ControlNet checkpoint.

        Args:
            prompt (str): A text description of the desired art piece.
            image_url (str): The URL of the image to be used as a reference for generation.
            controlnet_checkpoint (str): The ControlNet checkpoint path or identifier.
            seed (int, optional): The seed for random number generator.

        Returns:
            PIL.Image.Image: The generated art piece as an image.

        Raises:
            ValueError: If any of the inputs are invalid.
        """
        if not prompt or not image_url or not controlnet_checkpoint:
            raise ValueError("Input parameters cannot be empty.")

        # Load control image and setup
        image = load_image(image_url)
        processor = NormalBaeDetector.from_pretrained('lllyasviel/Annotators')
        control_image = processor(image)

        # Setup controlnet and pipeline
        controlnet = ControlNetModel.from_pretrained(controlnet_checkpoint, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # Set manual seed
        generator = torch.manual_seed(seed)

        # Generate art
        generated_image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

        return generated_image


# test_function_code --------------------


    def test_generate_art_from_text():
        print("Testing started.")

        # Test case with valid parameters
        print("Testing case [1/1] started.")
        sample_prompt = "A head full of roses"
        sample_image_url = 'https://example.com/image.png'
        sample_controlnet_checkpoint = 'lllyasviel/control_v11p_sd15_normalbae'
        result_image = generate_art_from_text(sample_prompt, sample_image_url, sample_controlnet_checkpoint)

        assert isinstance(result_image, Image.Image), f"Test case [1/1] failed: The result is not an image." 
        print("Testing finished.")


# call_test_function_line --------------------

test_generate_art_from_text()