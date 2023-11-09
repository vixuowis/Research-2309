def generate_normal_map(image_path: str, output_path: str = 'output_normal_map.png'):
    '''
    This function generates a normal map from a given object image.
    
    Parameters:
    image_path (str): The path to the input image.
    output_path (str): The path to save the output normal map. Default is 'output_normal_map.png'.
    
    Returns:
    None
    '''
    from PIL import Image
    from transformers import pipeline
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    from diffusers.utils import load_image
    import torch
    import numpy as np
    import cv2

    image = load_image(image_path).convert('RGB')
    depth_estimator = pipeline('depth-estimation', model='Intel/dpt-hybrid-midas')
    image_depth = depth_estimator(image)['predicted_depth'][0].numpy()

    # Preprocess the depth image
    image_depth -= np.min(image_depth)
    image_depth /= np.max(image_depth)
    bg_threhold = 0.4
    x = cv2.Sobel(image_depth, cv2.CV_32F, 1, 0, ksize=3)
    x[image_depth < bg_threhold] = 0
    y = cv2.Sobel(image_depth, cv2.CV_32F, 0, 1, ksize=3)
    y[image_depth < bg_threhold] = 0
    z = np.ones_like(x) * np.pi * 2.0
    image_normal = np.stack([x, y, z], axis=2)
    image_normal /= np.sum(image_normal**2.0, axis=2, keepdims=True)**0.5
    image_normal = (image_normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-normal', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    image_normal_map = Image.fromarray(image_normal)
    image_normal_map.save(output_path)