# requirements_file --------------------

!pip install -U Pillow, transformers, torch

# function_import --------------------

from transformers import Swin2SRForConditionalGeneration
import torch

# function_code --------------------

def enhance_image_resolution(low_res_image_path: str, save_path: str) -> None:
    """
    Enhance the resolution of a low-resolution image by upscaling it.
    
    :param low_res_image_path: Path to the low-resolution image file.
    :param save_path: Path where the enhanced image will be saved.
    """
    model = Swin2SRForConditionalGeneration.from_pretrained('conde/Swin2SR-lightweight-x2-64')
    
    # Load the low-resolution image and convert it to a tensor
    low_res_image = Image.open(low_res_image_path)
    low_res_tensor = torch.tensor(np.array(low_res_image)).unsqueeze(0)
    
    # Upscale the image using the model
    with torch.no_grad():
        upscaled_tensor = model(low_res_tensor)
    
    # Convert the tensor back to an image
    upscaled_image = Image.fromarray(upscaled_tensor.squeeze(0).numpy())
    upscaled_image.save(save_path)

# test_function_code --------------------

def test_enhance_image_resolution():
    print("Testing enhance_image_resolution function.")
    # Assuming we have a small low-resolution image for testing
    test_image_path = 'data/low_res_test_image.jpg'
    output_image_path = 'data/enhanced_test_image.jpg'

    print("Test case [1/1] started.")
    enhance_image_resolution(test_image_path, output_image_path)
    assert os.path.exists(output_image_path), f"Test case [1/1] failed: Enhanced image not found at {output_image_path}"
    print("Test case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_enhance_image_resolution()