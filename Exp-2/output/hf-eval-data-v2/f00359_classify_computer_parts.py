# function_import --------------------

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_computer_parts(user_uploaded_image_file_path):
    """
    Classify the computer part in the given image using a pre-trained Vision Transformer model.

    Args:
        user_uploaded_image_file_path (str): The file path of the user uploaded image.

    Returns:
        str: The predicted label of the computer part in the image.
    """
    # Create an instance of the ViTImageProcessor class
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    # Load the pre-trained Vision Transformer model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    # Load the image from the user
    image = Image.open(user_uploaded_image_file_path)
    # Preprocess the image
    inputs = processor(images=image, return_tensors='pt')
    # Run the preprocessed image through the model
    outputs = model(**inputs)
    logits = outputs.logits
    # Find the predicted class index
    predicted_class_idx = logits.argmax(-1).item()
    # Get the human-readable predicted label
    predicted_label = model.config.id2label[predicted_class_idx]
    return predicted_label

# test_function_code --------------------

def test_classify_computer_parts():
    """
    Test the classify_computer_parts function with a sample image.
    """
    # Use a sample image URL for testing
    image_url = 'https://example.com/sample.jpg'
    # Download the image
    response = requests.get(image_url, stream=True)
    response.raw.decode_content = True
    # Save the image to a temporary file
    with open('temp.jpg', 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    # Call the function with the temporary file
    result = classify_computer_parts('temp.jpg')
    # Check the result
    assert isinstance(result, str), 'The result should be a string.'

# call_test_function_code --------------------

test_classify_computer_parts()