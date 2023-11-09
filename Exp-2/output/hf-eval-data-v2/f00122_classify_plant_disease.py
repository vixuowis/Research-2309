# function_import --------------------

import clip
from PIL import Image

# function_code --------------------

def classify_plant_disease(image_path: str):
    """
    Classify the disease of a plant based on an image.

    Args:
        image_path (str): The path to the image of the plant.

    Returns:
        dict: A dictionary where the keys are the possible diseases ('healthy', 'pest-infested', 'fungus-infected', 'nutrient-deficient') and the values are the probabilities of the plant having each disease.
    """
    model, preprocess = clip.load('timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k')
    image = preprocess(Image.open(image_path))
    candidate_labels = ['healthy', 'pest-infested', 'fungus-infected', 'nutrient-deficient']
    logits = model(image.unsqueeze(0)).logits
    probs = logits.softmax(dim=-1)
    return {label: prob.item() for label, prob in zip(candidate_labels, probs.squeeze())}

# test_function_code --------------------

def test_classify_plant_disease():
    """
    Test the classify_plant_disease function.
    """
    result = classify_plant_disease('test_image.jpg')
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert set(result.keys()) == {'healthy', 'pest-infested', 'fungus-infected', 'nutrient-deficient'}, 'The keys of the result should be the possible diseases.'
    assert all(isinstance(value, float) for value in result.values()), 'The values of the result should be floats.'
    assert abs(sum(result.values()) - 1) < 0.01, 'The probabilities should sum up to approximately 1.'

# call_test_function_code --------------------

test_classify_plant_disease()