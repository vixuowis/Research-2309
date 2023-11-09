# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# function_code --------------------

def generate_text_description(image, text):
    """
    Generate textual descriptions for images and videos using the pre-trained GIT model.

    Args:
        image (tensor): The encoded image tensor.
        text (str): The input text.

    Returns:
        str: The generated text description.
    """
    # Load the pre-trained GIT model and tokenizer
    model = AutoModelForCausalLM.from_pretrained('microsoft/git-large-textcaps')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/git-large-textcaps')

    # Prepare the image and text inputs
    input_ids = tokenizer(text, return_tensors="pt", padding=True).input_ids
    prompt_length = len(input_ids[0])

    # Concatenate the image and text tokens
    input_ids = torch.cat([image, input_ids], dim=1)

    # Run the model to generate text description
    output = model.generate(input_ids, max_length=prompt_length + 20)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

# test_function_code --------------------

def test_generate_text_description():
    """
    Test the function generate_text_description.
    """
    # Prepare a sample image and text
    image = torch.rand(1, 3, 224, 224)  # A random image
    text = 'This is a test.'

    # Generate text description
    description = generate_text_description(image, text)

    # Check the output type and content
    assert isinstance(description, str), 'The output should be a string.'
    assert len(description) > 0, 'The output should not be empty.'

# call_test_function_code --------------------

test_generate_text_description()