# function_import --------------------

from PIL import Image
import torch
import clip

# function_code --------------------

def classify_plant_disease(image_path: str, candidate_labels: list) -> dict:
    """
    Classify the disease of a plant based on an image.

    Args:
        image_path (str): The path to the image of the plant.
        candidate_labels (list): A list of candidate class names for the plant issues.

    Returns:
        dict: A dictionary where keys are the candidate labels and values are the corresponding probabilities.
    """
    model, preprocess = clip.load('timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k')
    image = preprocess(Image.open(image_path))
    logits = model(image.unsqueeze(0)).logits
    probs = logits.softmax(dim=-1)
    classification_results = {label: prob.item() for label, prob in zip(candidate_labels, probs.squeeze())}
    return classification_results

# test_function_code --------------------

def test_classify_plant_disease():
    """
    Test the function classify_plant_disease.
    """
    image_path = 'https://placekitten.com/200/300'
    candidate_labels = ['healthy', 'pest-infested', 'fungus-infected', 'nutrient-deficient']
    classification_results = classify_plant_disease(image_path, candidate_labels)
    assert isinstance(classification_results, dict)
    assert set(candidate_labels) == set(classification_results.keys())
    assert all(isinstance(value, float) for value in classification_results.values())
    assert abs(sum(classification_results.values()) - 1) < 1e-6
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_plant_disease()