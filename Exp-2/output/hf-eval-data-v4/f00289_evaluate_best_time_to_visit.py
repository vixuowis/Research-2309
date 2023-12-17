# requirements_file --------------------

!pip install -U transformers Pillow

# function_import --------------------

from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

# function_code --------------------

def evaluate_best_time_to_visit(image_path):
    # Load the pre-trained model and processor
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')

    # Open the image file
    image = Image.open(image_path)
    texts = ["\u597D\u7684\u53C2\u89C2\u65F6\u95F4", "\u4E0D\u662F\u597D\u7684\u53C2\u89C2\u65F6\u95F4"]

    # Process the image and the associated texts
    inputs = processor(images=image, text=texts, return_tensors='pt')
    outputs = model(**inputs)

    # Extract the logits and calculate the probabilities
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()
    result = dict(zip(texts, probs[0]))

    # Determine whether it is a good time to visit or not
    good_time_to_visit = result['\u597D\u7684\u53C2\u89C2\u65F6\u95F4'] > result['\u4E0D\u662F\u597D\u7684\u53C2\u89C2\u65F6\u95F4']
    return good_time_to_visit, result


# test_function_code --------------------

def test_evaluate_best_time_to_visit():
    print("Testing started.")
    # Assume the sample dataset is locally available as images
    test_image = 'test_image.jpg'  # Placeholder path

    # Test case 1: Evaluate a specific image path
    print("Testing case [1/1] started.")
    good_time, evaluations = evaluate_best_time_to_visit(test_image)
    assert isinstance(good_time, bool), f"Test case [1/1] failed: Expected a boolean result, but got {type(good_time)}"
    assert '\u597D\u7684\u53C2\u89C2\u65F6\u95F4' in evaluations and '\u4E0D\u662F\u597D\u7684\u53C2\u89C2\u65F6\u95F4' in evaluations, f"Test case [1/1] failed: Evaluation object does not contain expected keys."
    print("Testing finished.")

    # Run the test function
    test_evaluate_best_time_to_visit()
