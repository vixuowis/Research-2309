# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# function_code --------------------

def generate_image_text_description(image, text):
    """
    Generate a textual description for a given image using a pre-trained GIT model.

    :param image: The input image to be described.
    :param text: (Optional) Additional text context to help generate the description.
    :return: A string containing the generated textual description.
    """
    model = AutoModelForCausalLM.from_pretrained('microsoft/git-large-textcaps')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/git-large-textcaps')

    # Encode text input and concatenate with encoded image tensor
    input_ids = tokenizer(text, return_tensors='pt', padding=True).input_ids
    prompt_length = len(input_ids[0])
    encoded_image = torch.tensor(image)  # Replace with actual image encoding logic
    input_ids = torch.cat([encoded_image, input_ids], dim=1)

    # Generate the textual description
    output = model.generate(input_ids, max_length=prompt_length + 20)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

# test_function_code --------------------

def test_generate_image_text_description():
    print("Testing started.")
    sample_image = [0]  # Replace with actual image loading logic
    sample_text = 'Describe this image:'

    # Test case 1: Check if the function returns a non-empty string
    print("Testing case [1/1] started.")
    description = generate_image_text_description(sample_image, sample_text)
    assert description, f"Test case [1/1] failed: The function returned an empty description"
    print("Testing finished.")

# Running the test function
test_generate_image_text_description()