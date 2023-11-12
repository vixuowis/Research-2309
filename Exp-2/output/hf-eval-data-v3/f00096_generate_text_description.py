# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# function_code --------------------

def generate_text_description(image, text):
    """
    Generate a textual description for a given image and text using the GIT model.

    Args:
        image (torch.Tensor): The encoded image tensor.
        text (str): The input text.

    Returns:
        str: The generated text description.

    Raises:
        ReadTimeout: If the model loading from Hugging Face Transformers times out.
    """
    # Load the pre-trained GIT model and tokenizer
    model = AutoModelForCausalLM.from_pretrained('microsoft/git-large-textcaps')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/git-large-textcaps')

    # Prepare the image and text inputs
    input_ids = tokenizer(text, return_tensors='pt', padding=True).input_ids
    prompt_length = len(input_ids[0])

    # Concatenate the image and text tokens
    input_ids = torch.cat([image, input_ids], dim=1)

    # Run the model to generate text description
    output = model.generate(input_ids, max_length=prompt_length + 20)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

# test_function_code --------------------

def test_generate_text_description():
    """
    Test the generate_text_description function.
    """
    # Prepare a dummy image tensor and text
    image = torch.rand(1, 3, 224, 224)
    text = 'A random text'

    # Call the function with the dummy inputs
    description = generate_text_description(image, text)

    # Assert the output is a string
    assert isinstance(description, str), 'The output should be a string.'

    # Assert the output is not empty
    assert description, 'The output should not be empty.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_text_description()