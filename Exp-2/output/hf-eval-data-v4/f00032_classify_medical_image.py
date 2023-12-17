# requirements_file --------------------

!pip install -U transformers pillow

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_medical_image(image_path):
    """
    Classify a medical image to determine whether it's an X-ray, an MRI scan, or a CT scan.

    Parameters:
    image_path (str): The file path to the medical image.

    Returns:
    str: The predicted class of the medical image.
    """
    # Load the zero-shot image classification model
    clip = pipeline('zero-shot-image-classification', model='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    # Define possible class names
    possible_class_names = ['X-ray', 'MRI scan', 'CT scan']
    # Classify the image
    result = clip(image_path, possible_class_names)
    # Get the class with the highest probability
    predicted_class = max(result, key=lambda x: x['scores'])['labels']
    return predicted_class

# test_function_code --------------------

def test_classify_medical_image():
    print("Testing started.")
    # Assuming a dummy image path (would normally be an image file)
    sample_image_path = 'dummy_image_path'

    # Testing classification
    print("Testing medical image classification started.")
    predicted_class = classify_medical_image(sample_image_path)
    assert predicted_class in ['X-ray', 'MRI scan', 'CT scan'], f"Test failed: Prediction {predicted_class} not among expected classes."
    print("Testing medical image classification completed successfully.")

# Run the test function
test_classify_medical_image()