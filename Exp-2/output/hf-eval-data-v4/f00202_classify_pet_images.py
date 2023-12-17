# requirements_file --------------------

!pip install -U PIL requests transformers

# function_import --------------------

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_pet_images(image_path, categories):
    """
    Classify an image of a pet into different categories using a pre-trained CLIP model.

    Parameters:
        image_path (str): The path to the image file.
        categories (list): A list of category descriptions (e.g., ['a photo of a cat', 'a photo of a dog']).

    Returns:
        dict: A dictionary with categories as keys and their corresponding probabilities as values.
    """
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    image = Image.open(image_path)
    inputs = processor(text=categories, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    category_probs = {category: float(probs[0][i]) for i, category in enumerate(categories)}
    return category_probs

# test_function_code --------------------

def test_classify_pet_images():
    print("Testing classify_pet_images started.")
    image_path = 'test_pet_image.jpg'
    categories = ['a photo of a cat', 'a photo of a dog']
    results = classify_pet_images(image_path, categories)
    assert isinstance(results, dict), "The result should be a dictionary."
    assert all(isinstance(value, float) for value in results.values()), "All probabilities should be float type."
    assert set(results.keys()) == set(categories), "Keys of the result should match the provided categories."
    print("Testing classify_pet_images finished successfully.")

# Run the test function
test_classify_pet_images()