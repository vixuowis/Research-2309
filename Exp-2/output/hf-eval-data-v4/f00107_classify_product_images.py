# requirements_file --------------------

!pip install -U transformers torch datasets

# function_import --------------------

from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
import torch
from datasets import load_dataset

# function_code --------------------

def classify_product_images(image_path):
    # Load the image dataset
    dataset = load_dataset('your_dataset_name', split='test')
    image = dataset[image_path]['image'][0]

    # Initialize the feature extractor and model
    feature_extractor = ConvNextFeatureExtractor.from_pretrained('facebook/convnext-large-224')
    model = ConvNextForImageClassification.from_pretrained('facebook/convnext-large-224')

    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Classify the image
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()

    # Return the predicted label
    return model.config.id2label[predicted_label]

# test_function_code --------------------

def test_classify_product_images():
    print('Testing image classification...')

    # Test sample image
    sample_image_path = 'sample_image.jpg'  # Path to a sample image
    predicted_label = classify_product_images(sample_image_path)

    # Expected output (assuming correct classification)
    expected_label = 'product_category_example'  # Example category

    # Test assertion
    assert predicted_label == expected_label, f'Image classification failed: {predicted_label} does not match {expected_label}'

    print('Test passed!')