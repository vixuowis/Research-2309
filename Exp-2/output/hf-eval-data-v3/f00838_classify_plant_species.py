# function_import --------------------

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

# function_code --------------------

def classify_plant_species(image_path):
    """
    Classify the species of plants in an image using a pre-trained model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The predicted class of the plant in the image.
    """
    image = Image.open(image_path)
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_plant_species():
    """
    Test the classify_plant_species function.
    """
    assert classify_plant_species('https://placekitten.com/200/300') == 'Predicted class: cat'
    assert classify_plant_species('https://placekitten.com/g/200/300') == 'Predicted class: cat'
    assert classify_plant_species('https://placekitten.com/200/300?image=0') == 'Predicted class: cat'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_plant_species()