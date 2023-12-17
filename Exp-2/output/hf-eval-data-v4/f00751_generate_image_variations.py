# requirements_file --------------------

!pip install -U diffusers torchvision pillow

# function_import --------------------

from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms

# function_code --------------------

def generate_image_variations(image_path, output_dir, num_variations=3, guidance_scale=7.5):
    # Load the original image
    original_image = Image.open(image_path)

    # Define the transformation for the image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    ])

    # Apply the transformation to the original image
    transformed_image = transform(original_image).unsqueeze(0)

    # Load the pre-trained model
    sd_pipeline = StableDiffusionImageVariationPipeline.from_pretrained('lambdalabs/sd-image-variations-diffusers', revision='v2.0')

    # Generate image variations
    output = sd_pipeline(transformed_image, guidance_scale=guidance_scale)

    # Save the generated image variations
    for i, variation in enumerate(output['images'][:num_variations]):
        variation.save(f'{output_dir}/variation_{i}.jpg')

    return [f'{output_dir}/variation_{i}.jpg' for i in range(num_variations)]

# test_function_code --------------------

def test_generate_image_variations():
    print("Testing started.")
    image_path = 'path/to/sample_image.jpg' # replace with path to a sample image
    output_dir = 'path/to/output' # replace with path to the output directory
    num_variations = 5
    guidance_scale = 7.5

    # Generate image variations and get the list of saved file paths
    generated_files = generate_image_variations(image_path, output_dir, num_variations, guidance_scale)

    # Check if the correct number of files have been created
    print("Testing image variation generation [1/1] started.")
    assert len(generated_files) == num_variations, f"Test failed: Expected {num_variations} generated images, got {len(generated_files)}"
    print("Test passed.")
    print("Testing finished.")

# Run the test function
test_generate_image_variations()