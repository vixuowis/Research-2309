# requirements_file --------------------

!pip install -U diffusers Pillow

# function_import --------------------

from diffusers import DiffusionPipeline

# function_code --------------------

def generate_face_image(num_inference_steps=200):
    «
    Generates a high-quality image of a face using the Latent Diffusion Model.

    Args:
        num_inference_steps (int): The number of inference steps to use during image generation.

    Returns:
        A PIL image object of the generated face.

    Raises:
        RuntimeError: If loading the pre-trained model fails or inference fails.
    »
    model_id = 'CompVis/ldm-celebahq-256'
    pipeline = DiffusionPipeline.from_pretrained(model_id)
    image = pipeline(num_inference_steps=num_inference_steps)[0]
    return image

# test_function_code --------------------

def test_generate_face_image():
    print("Testing started.")

    # Test case 1: Check correct object type
    print("Testing case [1/3] started.")
    generated_image = generate_face_image()
    assert isinstance(generated_image, Image.Image), f"Test case [1/3] failed: Expected a PIL Image object, but got {type(generated_image)}"

    # Test case 2: Check with different inference steps
    print("Testing case [2/3] started.")
    generated_image = generate_face_image(num_inference_steps=100)
    assert isinstance(generated_image, Image.Image), f"Test case [2/3] failed: Expected a PIL Image object, but got {type(generated_image)}"

    # Test case 3: Check that function runs without errors
    print("Testing case [3/3] started.")
    try:
        generate_face_image()
        assert True, "Test case [3/3] passed."
    except Exception as e:
        assert False, f"Test case [3/3] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_face_image()