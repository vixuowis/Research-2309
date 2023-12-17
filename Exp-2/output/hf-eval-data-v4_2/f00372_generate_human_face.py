# requirements_file --------------------

!pip install -U diffusers

# function_import --------------------

from diffusers import DiffusionPipeline

# function_code --------------------

def generate_human_face(model_id: str) -> Image.Image:
    """
    Generates a human face image using a pre-trained diffusion model.

    Args:
        model_id: A string representing the pre-trained model identifier.

    Returns:
        An Image object representing the generated human face.

    Raises:
        RuntimeError: If the pre-trained model cannot be loaded.

    """
    try:
        pipeline = DiffusionPipeline.from_pretrained(model_id)
        image = pipeline().images[0]
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_id}: {str(e)}")

# test_function_code --------------------

def test_generate_human_face():
    print("Testing started.")
    model_id = 'google/ncsnpp-ffhq-256'

    # Test case 1: Check if the generated image is not None
    print("Testing case [1/1] started.")
    image = generate_human_face(model_id)
    assert image is not None, f"Test case [1/1] failed: Generated image is None."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_human_face()