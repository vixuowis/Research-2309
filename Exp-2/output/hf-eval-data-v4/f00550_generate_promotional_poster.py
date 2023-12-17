# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_promotional_poster(prompt, negative_prompt):
    # Initialize the text-to-image model pipeline from Hugging Face
    model = pipeline('text-to-image', model='SG161222/Realistic_Vision_V1.4')

    # Generate the image based on provided prompts
    result = model(prompt, negative_prompt=negative_prompt)
    return result

# test_function_code --------------------

def test_generate_promotional_poster():
    print("Testing started.")

    # Define positive and negative prompts
    prompt = "A promotional poster for a new line of summer clothing featuring happy people wearing the clothes, with a sunny beach background, clear blue sky, and palm trees. Image dimensions should be poster-sized, high-resolution, and vibrant colors."
    negative_prompt = "winter, snow, cloudy, low-resolution, dull colors, indoor, mountain"

    # Call the function to generate the image
    result = generate_promotional_poster(prompt, negative_prompt)

    # Test case: Check if the result is not empty
    print("Testing case [1/1] started.")
    assert result, f"Test case [1/1] failed: Expected a non-empty result, got {result}"
    print("Testing finished.")

# Run the test function
test_generate_promotional_poster()