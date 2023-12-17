# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# function_code --------------------

def generate_image_text_description(image, text):
    """
    Generate a textual description for a given image with an optional initial text prompt.

    Args:
        image (Tensor): A tensor representing the encoded image.
        text (str): Initial text to concatenate with the image encoding. Optional.

    Returns:
        str: Generated text based on the given image and initial text.

    Raises:
        ValueError: If the image is not properly encoded.
    """
    tokenizer = AutoTokenizer.from_pretrained('microsoft/git-large-textcaps')
    model = AutoModelForCausalLM.from_pretrained('microsoft/git-large-textcaps')
    # Prepare the image and text inputs
    input_ids = tokenizer(text, return_tensors="pt", padding=True).input_ids
    prompt_length = len(input_ids[0])
    if not isinstance(image, torch.Tensor):
        raise ValueError("The image must be a pre-encoded tensor.")
    input_ids = torch.cat([image, input_ids], dim=1)
    # Generate text description
    output = model.generate(input_ids, max_length=prompt_length + 20)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# test_function_code --------------------

def test_generate_image_text_description():
    print("Testing started.")
    # Assuming `load_image_tensor` function is defined elsewhere for loading and encoding a sample image
    sample_image = load_image_tensor('sample.jpg')
    sample_text = 'A small cat'

    # Testing case 1: Correct image tensor and text
    print("Testing case [1/2] started.")
    result = generate_image_text_description(sample_image, sample_text)
    assert isinstance(result, str), "Test case [1/2] failed: Result is not a string."

    # Testing case 2: Incorrect image input type
    print("Testing case [2/2] started.")
    try:
        generate_image_text_description('not a tensor', sample_text)
        assert False, "Test case [2/2] failed: ValueError not raised for incorrect image type."
    except ValueError:
        pass  # Expected
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_image_text_description()