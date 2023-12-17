# requirements_file --------------------

!pip install -U transformers torch datasets

# function_import --------------------

from transformers import AutoFeatureExtractor, RegNetForImageClassification
import torch

# function_code --------------------

def classify_security_camera_image(image):
    """
    Classify an image taken by a security camera using a pretrained RegNet model.

    Parameters:
    image: a PIL Image or a NumPy array representing the image to classify.

    Returns:
    The label of the image predicted by the model.
    """
    feature_extractor = AutoFeatureExtractor.from_pretrained('zuppif/regnet-y-040')
    model = RegNetForImageClassification.from_pretrained('zuppif/regnet-y-040')
    inputs = feature_extractor(image, return_tensors='pt')

    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]

# test_function_code --------------------

def test_classify_security_camera_image():
    from datasets import load_dataset
    from PIL import Image
    print("Testing started.")
    dataset = load_dataset('huggingface/cats-image')
    # Assuming the dataset has 'image' and 'label' columns
    sample_data = dataset['test'][0]
    image, label = Image.open(sample_data['image']), sample_data['label']

    predicted_label = classify_security_camera_image(image)

    assert predicted_label == label, f"Test failed: Expected {label}, got {predicted_label}"
    print("Testing finished.")

# Run the test function
test_classify_security_camera_image()