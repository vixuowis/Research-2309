# requirements_file --------------------

!pip install -U transformers==4.0.0 torch==1.9.0 Pillow==8.3.2

# function_import --------------------

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

# function_code --------------------

def classify_computer_part(image_path):
    # Create processor and model instances from pre-trained data.
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Open the image file and process it.
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors='pt')

    # Run the model on the processed image to classify it.
    outputs = model(**inputs)
    logits = outputs.logits

    # Find the predicted class index and corresponding label.
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]

    return predicted_label

# test_function_code --------------------

def test_classify_computer_part():
    print("Testing started.")

    # Test case: Correctly identifies a known computer part in image.
    known_part_image_path = 'sample_images/known_computer_part.jpg'
    expected_label = 'computer_part_label'
    print("Testing known computer part image.")
    predicted_label = classify_computer_part(known_part_image_path)
    assert predicted_label == expected_label, f"Test failed: predicted {{predicted_label}} instead of {{expected_label}}."

    # Additional test cases can be added as needed.

    print("Testing finished.")