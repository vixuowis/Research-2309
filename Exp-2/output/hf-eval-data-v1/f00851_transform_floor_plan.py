def transform_floor_plan(input_image_path: str, output_image_path: str, num_inference_steps: int = 20):
    """
    Transforms a floor plan image into a simplified straight line drawing using a pretrained ControlNet model.

    Args:
        input_image_path (str): Path to the input floor plan image.
        output_image_path (str): Path to save the output transformed image.
        num_inference_steps (int, optional): Number of inference steps to process the image. Default is 20.

    Returns:
        None

    Raises:
        FileNotFoundError: If the input_image_path does not exist.
    """
    from PIL import Image
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    import torch
    from controlnet_aux import MLSDdetector
    from diffusers.utils import load_image

    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f'Input image {input_image_path} not found.')

    mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    floor_plan_img = load_image(input_image_path)
    floor_plan_img = mlsd(floor_plan_img)
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-mlsd', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    result_img = pipe(floor_plan_img, num_inference_steps=num_inference_steps).images[0]
    result_img.save(output_image_path)