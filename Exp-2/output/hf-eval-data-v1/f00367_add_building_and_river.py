def add_building_and_river(image_path: str, output_path: str):
    '''
    This function adds a building and a river to a given landscape image using the ControlNetModel from Hugging Face.
    
    Parameters:
    image_path (str): The path to the input landscape image.
    output_path (str): The path to save the output image.
    
    Returns:
    None
    '''
    from PIL import Image
    import torch
    from diffusers.utils import load_image
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

    control_image = load_image(image_path).convert('RGB')
    prompt = "add a building and a river"

    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11e_sd15_ip2p', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)

    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    image.save(output_path)