def generate_image_from_text(prompt, control_image_path):
    '''
    This function generates an image from a text description using a pretrained ControlNet model.
    The function takes a text prompt and a path to a control image as input, and returns the path to the generated image.
    '''
    import torch
    from diffusers.utils import load_image
    from PIL import Image
    from controlnet_aux import HEDdetector
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

    # Load the controlnet pretrained model
    checkpoint = 'lllyasviel/control_v11p_sd15_scribble'
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)

    # Create a pipeline using the pretrained StableDiffusionControlNetPipeline and the loaded controlnet model
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # Load the input scribble image
    scribble_image = Image.open(control_image_path)

    # Generate the output image
    generator = torch.manual_seed(0)
    image = pipe(prompt, num_inference_steps=30, generator=generator, image=scribble_image).images[0]

    # Save the generated output image
    output_image_path = 'images/image_out.png'
    image.save(output_image_path)

    return output_image_path