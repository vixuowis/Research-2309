from diffusers import DDPMPipeline
import os


def generate_insect_image(model_name: str, output_path: str) -> None:
    """
    This function generates an insect image using a pretrained model from Hugging Face Transformers.

    Parameters:
    model_name (str): The name of the pretrained model to use for image generation.
    output_path (str): The path where the generated image will be saved.

    Returns:
    None
    """
    # Import the DDPMPipeline class from the diffusers package
    # Load the pretrained model
    pipeline = DDPMPipeline.from_pretrained(model_name)

    # Generate a new image using the loaded pipeline
    generated_image = pipeline().images[0]

    # Save the generated image to the specified output path
    generated_image.save(output_path)

    print(f'Image saved at {output_path}')