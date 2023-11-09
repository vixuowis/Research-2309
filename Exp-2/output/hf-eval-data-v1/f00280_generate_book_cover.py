def generate_book_cover(input_image_path: str, prompt: str, output_image_path: str = 'images/image_out.png'):
    '''
    This function generates a book cover based on a given prompt and an input image.
    The function uses the ControlNetModel from Hugging Face to generate the image.
    
    Args:
    input_image_path (str): The path to the input image.
    prompt (str): The prompt describing the image to generate.
    output_image_path (str, optional): The path to save the generated image. Defaults to 'images/image_out.png'.
    
    Returns:
    None
    '''
    from PIL import Image
    import torch
    from controlnet_aux import NormalBaeDetector
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

    image = Image.open(input_image_path)
    processor = NormalBaeDetector.from_pretrained('lllyasviel/Annotators')
    control_image = processor(image)

    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_normalbae', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(33)
    generated_image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    generated_image.save(output_image_path)