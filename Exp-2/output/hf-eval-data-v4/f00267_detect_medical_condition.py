# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# function_code --------------------


def detect_medical_condition(image, question):
    """
    Detects the medical condition in the given image using a question.

    Parameters:
        image (str): The file path or URL to the image.
        question (str): The question asking about the medical condition present in the image.

    Returns:
        str: The detected medical condition.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained('microsoft/git-large-textvqa')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/git-large-textvqa')
    encoded_input = tokenizer(question, image, return_tensors='pt')
    generated_tokens = model.generate(**encoded_input)
    detected_condition = tokenizer.decode(generated_tokens[0])
    return detected_condition


# test_function_code --------------------


def test_detect_medical_condition():
    print("Testing started.")
    # Load image and define question for testing purpose
    image = 'path/to/sample/image.jpg'  # Replace with an actual image path or URL
    question = 'What medical condition is present in the image?'

    # Call the function with the test image and question
    detected_condition = detect_medical_condition(image, question)

    # Check if a condition is returned and not an empty string
    assert detected_condition, "Detection failed: No condition returned."
    print("Testing finished successfully with detected condition:", detected_condition)

# Run the test function
test_detect_medical_condition()
