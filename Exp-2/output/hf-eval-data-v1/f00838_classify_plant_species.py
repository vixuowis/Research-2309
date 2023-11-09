from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image


def classify_plant_species(image_path):
    """
    Classify the species of plants in an image using a pre-trained Vision Transformer model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The predicted class of the plant in the image.
    """
    # Load the image data from the specified file
    image = Image.open(image_path)

    # Load the pre-trained Vision Transformer model and the associated image processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Process the image and run it through the model
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)

    # Extract the predicted class index from the model's output
    predicted_class_idx = outputs.logits.argmax(-1).item()

    # Return the predicted class
    return model.config.id2label[predicted_class_idx]