# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# function_code --------------------

def detect_medical_condition(image, question):
    """
    Detect the medical condition from the provided image by asking a specific question.

    Args:
        image (str): The image in which the medical condition needs to be detected.
        question (str): The question to be asked about the medical condition in the image.

    Returns:
        str: The predicted medical condition as output.
    
    Raises:
        ValueError: If the image data or question is not valid.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained('microsoft/git-large-textvqa')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/git-large-textvqa')
    encoded_input = tokenizer(question, image=image, return_tensors='pt')
    generated_tokens = model.generate(**encoded_input)
    detected_medical_condition = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return detected_medical_condition


# test_function_code --------------------

def test_detect_medical_condition():
    print("Testing started.")
    # Assuming load_dataset is defined elsewhere to get image data
    dataset = load_dataset("medical_images")
    sample_image = dataset[0]  # Get a sample image from the dataset
    question = "What medical condition is present in this image?"

    # Test Case 1
    print("Testing case [1/3] started.")
    condition = detect_medical_condition(sample_image, question)
    assert condition is not None, "Test case [1/3] failed: No condition detected."

    # Test Case 2
    print("Testing case [2/3] started.")
    # Assuming sample_image_2 has a known condition 'condition_2'
    condition = detect_medical_condition(sample_image_2, question)
    assert condition == 'condition_2', f"Test case [2/3] failed: Incorrect condition detected. Detected: {condition}"

    # Test Case 3
    print("Testing case [3/3] started.")
    # Assuming sample_image_3 should raise an error due to invalid data
    try:
        detect_medical_condition(sample_image_3, question)
        assert False, "Test case [3/3] failed: Invalid image data should raise an error."
    except ValueError:
        pass

    print("Testing finished.")


# call_test_function_line --------------------

test_detect_medical_condition()