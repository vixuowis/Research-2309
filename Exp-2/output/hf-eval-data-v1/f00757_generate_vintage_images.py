from diffusers import DDPMPipeline


def generate_vintage_images():
    """
    Generate vintage images using a pretrained model.

    This function uses the DDPMPipeline from diffusers and a pretrained model
    'pravsels/ddpm-ffhq-vintage-finetuned-vintage-3epochs' from Hugging Face Transformers.
    The model is a Denoising Diffusion Probabilistic Model fine-tuned on 3 epochs of vintage images.

    Returns:
        generated_images: A list of generated images.
    """
    pipeline = DDPMPipeline.from_pretrained('pravsels/ddpm-ffhq-vintage-finetuned-vintage-3epochs')
    generated_images = pipeline().images
    return generated_images