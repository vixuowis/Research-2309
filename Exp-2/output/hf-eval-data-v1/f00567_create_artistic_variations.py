from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode, Normalize

def create_artistic_variations(image_path):
    '''
    This function creates artistic variations of an input image using the StableDiffusionImageVariationPipeline from the diffusers library.
    Args:
    image_path (str): The path to the input image.
    Returns:
    str: The path to the output image.
    '''
    # Create an instance of the StableDiffusionImageVariationPipeline by loading the pre-trained model
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained('lambdalabs/sd-image-variations-diffusers', revision='v2.0')
    # Load the input image
    im = Image.open(image_path)
    # Create a set of transforms to preprocess the input image for the model
    tform = Compose([
        ToTensor(),
        Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=False),
        Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])
    # Apply transforms to the input image
    inp = tform(im).unsqueeze(0)
    # Pass the input image to the pipeline
    out = sd_pipe(inp, guidance_scale=3)
    # Save the output image
    out['images'][0].save('result.jpg')
    return 'result.jpg'