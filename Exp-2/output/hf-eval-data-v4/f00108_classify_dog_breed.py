# requirements_file --------------------

!pip install -U transformers==4.0.0 torch==1.9.0 Pillow==8.3.2 requests==2.26.0

# function_import --------------------

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_dog_breed(image_url):
    # Load the image from the URL
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # Initialize the processor and model with the pre-trained weights
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    
    # Preprocess the image and prepare the inputs for the model
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Get the predicted class index and the corresponding label
    predicted_class_idx = logits.argmax(-1).item()
    predicted_breed = model.config.id2label[predicted_class_idx]
    
    return predicted_breed

# test_function_code --------------------

def test_classify_dog_breed():
    print('Testing classify_dog_breed function...')
    
    # Test case: A known dog image URL
    test_image_url = 'https://example.com/known_dog_breed.jpg'
    expected_breed = 'golden_retriever'
    
    predicted_breed = classify_dog_breed(test_image_url)
    assert predicted_breed == expected_breed, f'Failed on known dog breed. Expected: {expected_breed}, got: {predicted_breed}'
    
    print('All tests passed!')