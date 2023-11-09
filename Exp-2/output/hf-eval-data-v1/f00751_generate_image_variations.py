from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms

def generate_image_variations(image_path: str, output_path: str = 'result.jpg', guidance_scale: int = 3):
    """
    Generate different variations of a product image using a pre-trained model.

    Args:
        image_path (str): Path to the original image.
        output_path (str, optional): Path to save the generated image. Defaults to 'result.jpg'.
        guidance_scale (int, optional): Controls the number and style of variations. Defaults to 3.

    Returns:
        None. The function saves the generated image to the specified output path.
    """
    # Load the original image
    image = Image.open(image_path)

    # Load the pre-trained model
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained('lambdalabs/sd-image-variations-diffusers', revision='v2.0')

    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    ])

    # Apply the transformation to the image and add a batch dimension
    inp = transform(image).unsqueeze(0)

    # Generate the image variations
    output = sd_pipe(inp, guidance_scale=guidance_scale)

    # Save the generated image
    output['images'][0].save(output_path)