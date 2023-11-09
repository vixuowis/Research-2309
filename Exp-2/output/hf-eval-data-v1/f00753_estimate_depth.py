def estimate_depth(input_image_path: str, output_image_path: str) -> None:
    """
    This function estimates the depth map from an input image of a street filled with people.
    It uses the 'lllyasviel/sd-controlnet-depth' model from Hugging Face.

    Args:
        input_image_path (str): The path to the input image.
        output_image_path (str): The path where the output image will be saved.

    Returns:
        None
    """
    from transformers import pipeline
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    from PIL import Image
    import numpy as np
    import torch
    from diffusers.utils import load_image

    depth_estimator = pipeline('depth-estimation')
    input_image = load_image(input_image_path)
    depth_image = depth_estimator(input_image)['depth']

    # Save the output
    depth_image_array = np.array(depth_image)
    depth_image_array = depth_image_array[:, :, None] * np.ones(3, dtype=np.float32)[None, None, :]
    output_image = Image.fromarray(depth_image_array.astype(np.uint8))
    output_image.save(output_image_path)