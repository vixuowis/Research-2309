# requirements_file --------------------

!pip install -U transformers torch torchvision pillow

# function_import --------------------

from transformers import AutoModelForImageClassification
import torch
from PIL import Image
from torchvision import transforms

# function_code --------------------

def classify_social_media_images(image_paths):
    # Load the pretrained image classification model
    model = AutoModelForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224-bottom_cleaned_data')
    model.eval()  # Set the model to evaluation mode

    # Define the image transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Classify each image
    results = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image_tensor)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            top_category = torch.argmax(predictions, dim=-1)
            results.append(top_category.item())

    return results

# test_function_code --------------------

def test_classify_social_media_images():
    print('Testing classify_social_media_images function.')
    # Assuming 'load_test_images' is a hypothetical function to load test images
    test_images, expected_results = load_test_images()

    # Get the classification results
    actual_results = classify_social_media_images(test_images)

    # Assert that the expected results match the actual results
    assert actual_results == expected_results, 'Classification results do not match the expected results.'

    print('All tests passed.')