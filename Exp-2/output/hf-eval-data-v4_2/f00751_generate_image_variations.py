# requirements_file --------------------

!pip install -U diffusers torchvision Pillow

# function_import --------------------

from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
import torchvision.transforms as transforms

# function_code --------------------

def generate_image_variations(image_path, save_folder, num_variations=5, guidance_scale=7.5):
    """Generate variations of an image using a pre-trained Stable Diffusion model.

    Args:
        image_path (str): Path to the original image file.
        save_folder (str): Folder path where the image variations will be saved.
        num_variations (int): Number of image variations to generate. Default is 5.
        guidance_scale (float): Controls the 'creativity' of the variations. Default is 7.5.

    Returns:
        list[str]: List of paths to the saved image variations.

    Raises:
        FileNotFoundError: If the original image file is not found.
        OSError: If there is an issue saving the image variations.
    """
    image = Image.open(image_path)

    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained('lambdalabs/sd-image-variations-diffusers', revision='v2.0')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    ])
    inp = transform(image).unsqueeze(0)
    output = sd_pipe(inp, guidance_scale=guidance_scale)

    saved_paths = []
    for i, variation in enumerate(output['images'][:num_variations]):
        save_path = f'{save_folder}/{i+1}.jpg'
        variation.save(save_path)
        saved_paths.append(save_path)

    return saved_paths

# test_function_code --------------------

def test_generate_image_variations():
    print("Testing started.")

    # Test case 1: Correct input, expect list of image paths
    print("Testing case [1/1] started.")
    image_variations = generate_image_variations('valid_image_path.jpg', 'output_folder', 3, 5.0)
    assert len(image_variations) == 3 and all(isinstance(path, str) for path in image_variations), f"Test case [1/1] failed: Expected 3 image paths, got {len(image_variations)}"

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_image_variations()