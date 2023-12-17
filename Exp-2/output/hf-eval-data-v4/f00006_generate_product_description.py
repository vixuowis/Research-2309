# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import GenerativeImage2TextModel

# function_code --------------------

def generate_product_description(product_image):
    """
    Generate a descriptive text of a product based on its image using a pre-trained model.

    Parameters:
        product_image (PIL.Image): An image object of the product.

    Returns:
        str: A descriptive text generated for the product.
    """
    # Load pre-trained Generative Image-to-Text Model from Hugging Face Transformers
    git_model = GenerativeImage2TextModel.from_pretrained('microsoft/git-large-coco')
    
    # Generate descriptive text for the product image
    product_description = git_model.generate_image_description(product_image)
    return product_description


# test_function_code --------------------

def test_generate_product_description():
    print("Testing started.")
    # Assuming we have a function to load a sample image
    sample_image = load_sample_image()

    # Testing case: Generate product description
    print("Testing product description generation.")
    description = generate_product_description(sample_image)
    assert isinstance(description, str), f"Expected a string description, got: {type(description)}"
    assert len(description) > 0, "Generated description is empty."
    print("Testing finished.")

# Helper function to load a sample image
# This would be replaced by actual code to load an image in a real scenario
def load_sample_image():
    from PIL import Image
    import requests
    from io import BytesIO
    # Sample image URL
    url = "https://example.com/sample_product_image.jpg"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

# Run the test
test_generate_product_description()
