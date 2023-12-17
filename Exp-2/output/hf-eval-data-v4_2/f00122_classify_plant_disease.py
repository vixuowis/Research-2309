# requirements_file --------------------

!pip install -U clip openai transformers pillow torchvision

# function_import --------------------

import clip
from PIL import Image

# function_code --------------------

def classify_plant_disease(image_path, model_name='timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k'):
    """Classify a plant disease from an image.

    Args:
        image_path (str): The file path to the image of the plant.
        model_name (str): The name of the pre-trained model to use for classification.

    Returns:
        dict: A dictionary with class names as keys and their respective probabilities as values.
    
    Raises:
        FileNotFoundError: If the image_path does not exist.
        RuntimeError: If there's an error loading the model or processing the image.
    """
    model, preprocess = clip.load(model_name)
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    candidate_labels = ['healthy', 'pest-infested', 'fungus-infected', 'nutrient-deficient']
    logits = model(image).logits
    probs = logits.softmax(dim=-1)
    return {label: prob.item() for label, prob in zip(candidate_labels, probs.squeeze())}

# test_function_code --------------------

from torchvision.datasets import CIFAR10
from torchvision import transforms

def test_classify_plant_disease():
    print("Testing started.")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    sample_image, _ = dataset[0]  # Use the first image from the CIFAR10 dataset as a sample

    # Testing case 1: Health classification
    print("Testing case [1/3] started.")
    result = classify_plant_disease('sample_image.jpg')
    assert 'healthy' in result, f"Test case [1/3] failed: 'healthy' not found in the result {result}"

    # Testing case 2: Pests classification
    print("Testing case [2/3] started.")
    result = classify_plant_disease('sample_image.jpg')
    assert 'pest-infested' in result, f"Test case [2/3] failed: 'pest-infested' not found in the result {result}"

    # Testing case 3: Fungi classification
    print("Testing case [3/3] started.")
    result = classify_plant_disease('sample_image.jpg')
    assert 'fungus-infected' in result, f"Test case [3/3] failed: 'fungus-infected' not found in the result {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_plant_disease()