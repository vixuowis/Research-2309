# requirements_file --------------------

!pip install -U PIL transformers

# function_import --------------------

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_pet_images(image_path, model_name='openai/clip-vit-large-patch14'):
    # Load the pre-trained CLIP model and the processor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Open the image file
    image = Image.open(image_path)
    
    # Preprocess the inputs
    inputs = processor(text=['a photo of a cat', 'a photo of a dog'], images=image, return_tensors='pt', padding=True)
    
    # Get the model predictions
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    
    # Calculate the probabilities
    probs = logits_per_image.softmax(dim=1)
    
    # Return the probabilities
    return probs

# test_function_code --------------------

def test_classify_pet_images():
    print("Testing started.")
    sample_image_path = 'sample_image.jpg'  # Assume we have a sample image

    # Testing case [1/1] started
    print("Testing case [1/1] started.")
    probs = classify_pet_images(sample_image_path)
    assert probs.dim() == 2 and probs.size(1) == 2, f"Test case [1/1] failed: Expected probability tensor to have shape (*, 2) but got {probs.size()}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_pet_images()