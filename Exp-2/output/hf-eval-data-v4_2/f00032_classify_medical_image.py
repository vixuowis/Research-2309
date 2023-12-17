# requirements_file --------------------

!pip install -U transformers torch torchvision

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_medical_image(image_path: str) -> str:
    """
    Classify a medical image as an X-ray, MRI scan, or CT scan using
    zero-shot image classification with a pre-trained model.

    Args:
        image_path: A string path to the image file to be classified.

    Returns:
        A string with the highest probability class label from ['X-ray', 'MRI scan', 'CT scan'].

    Raises:
        FileNotFoundError: If the image_path does not exist.
        ValueError: If the classification model cannot process the image.
    """
    clip = pipeline('zero-shot-image-classification', model='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    possible_class_names = ['X-ray', 'MRI scan', 'CT scan']
    try:
        result = clip(image_path, possible_class_names)
        predicted_class = max(result['scores'], key=lambda x: x['score'])['label']
        return predicted_class
    except Exception as e:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"The image file at {image_path} doesn't exist.") from e
        else:
            raise ValueError("Error processing the image.") from e

# test_function_code --------------------

def test_classify_medical_image():
    print("Testing started.")
    
    # Test case 1: Image file exists and is classified
    print("Testing case [1/1] started.")
    test_image_path = 'test_data/medical_image_xray.png'  # This should be changed to an actual test image path
    classification_result = classify_medical_image(test_image_path)
    assert classification_result in ['X-ray', 'MRI scan', 'CT scan'], f"Test case [1/1] failed: Unexpected result {classification_result}"

# call_test_function_line --------------------

test_classify_medical_image()