# requirements_file --------------------

!pip install -U huggingface_hub, openai, transformers

# function_import --------------------

import clip
from PIL import Image
from huggingface_hub import hf_hub_download

# function_code --------------------

def classify_plant_disease(image_path, candidate_labels):
    # Import the necessary libraries
    import clip
    from PIL import Image

    # Load the pre-trained model from the Hugging Face Hub
    model, preprocess = clip.load('timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k')

    # Preprocess the image
    image = preprocess(Image.open(image_path))

    # Replace 'candidate_labels' with the labels of the plant diseases you expect
    logits = model(image.unsqueeze(0)).logits
    probs = logits.softmax(dim=-1)

    # Pair each label with its prediction probability and create a dictionary
    classification_results = {label: prob.item() for label, prob in zip(candidate_labels, probs.squeeze())}
    return classification_results

# test_function_code --------------------

def test_classify_plant_disease():
    print("Testing classify_plant_disease function.")
    # Here, we would typically load an image or a set of images to test
    # For the purpose of this test, assume this is the path to a test image file
    test_image_path = hf_hub_download(repo_id='test_repo', filename='test_plant_image.jpg')
    candidate_labels = ['healthy', 'pest-infested', 'fungus-infected', 'nutrient-deficient']
    # Call the classification function on the test image
    results = classify_plant_disease(test_image_path, candidate_labels)
    # Perform the test - we expect to receive a dictionary with probabilities
    assert isinstance(results, dict), f'Expected dictionary of results, got {type(results)}'

    print("Test successful.")

# Run the test function
test_classify_plant_disease()