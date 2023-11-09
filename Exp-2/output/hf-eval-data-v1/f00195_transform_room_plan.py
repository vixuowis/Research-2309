import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch

def transform_room_plan(image_path):
    '''
    This function takes in the path of an image of a room plan and returns a better visual representation of the room plan.
    It uses the ControlNetModel from Hugging Face to perform the transformation.
    '''
    # Load room plan image
    image = load_image(image_path)
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    # Create ControlNetModel and pipeline
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-canny', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)

    # Process and save output
    transformed_image = pipe('room_plan', image, num_inference_steps=20).images[0]
    transformed_image.save('room_plan_transformed.png')
    return transformed_image