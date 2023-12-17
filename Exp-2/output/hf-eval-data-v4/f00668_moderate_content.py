# requirements_file --------------------

!pip install -U PIL requests transformers

# function_import --------------------

from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# function_code --------------------

def moderate_content(image_url, labels, model_name='OFA-Sys/chinese-clip-vit-large-patch14-336px'):
    # Load the pre-trained model and processor
    model = ChineseCLIPModel.from_pretrained(model_name)
    processor = ChineseCLIPProcessor.from_pretrained(model_name)
    
    # Load the image data
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # Process the image and text for the model
    inputs = processor(images=image, text=labels, return_tensors="pt", padding=True)
    
    # Perform the zero-shot classification
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    
    # Turn logits into probabilities
    probs = logits_per_image.softmax(dim=1)
    
    # Get the highest probability label
    max_idx = probs.argmax()
    result_label = labels[max_idx]

    return result_label, probs[0,max_idx].item()

# test_function_code --------------------

def test_moderate_content():
    print("Testing moderate_content started.")
    # Example test case from CIFAR100 dataset
    image_url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
    labels = ['safe content', 'unsafe content']

    # Call the function
    result_label, confidence = moderate_content(image_url, labels)
    print(f'Result: {result_label}, Confidence: {confidence}')
    assert result_label in labels, "Test case failed: The result label is not among the provided labels"
    assert confidence >= 0 and confidence <= 1, "Test case failed: Confidence score is not within the range [0, 1]"

    print("Testing moderate_content finished.")

# Run the test function
test_moderate_content()