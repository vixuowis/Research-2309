from transformers import AutoModelForImageClassification
import torch
from PIL import Image
from torchvision import transforms


def classify_images(image_paths, categories):
    """
    Classify images into various categories using a pre-trained model from Hugging Face Transformers.

    Args:
        image_paths (list of str): List of paths to the images to be classified.
        categories (list of str): List of categories that the images can be classified into.

    Returns:
        dict: A dictionary where the keys are the image paths and the values are the predicted categories.
    """
    # Load the pre-trained model
    model = AutoModelForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224-bottom_cleaned_data')

    # Define the transformation
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # Classify each image
    results = {}
    for image_path in image_paths:
        # Load and transform the image
        image = Image.open(image_path)
        image = transform(image)
        image = image.unsqueeze(0)

        # Make the prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        # Map the prediction to the corresponding category
        results[image_path] = categories[predicted.item()]

    return results