# requirements_file --------------------

!pip install -U transformers datasets

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_insurance_info(image_path, questions):
    """
    Extracts information from an insurance policy document image using Document-question-answering pipeline model.

    Args:
        image_path (str): The file path to the insurance document image.
        questions (list of str): A list of questions to retrieve information from the document.

    Returns:
        dict: A dictionary containing questions and their respective answers.

    Raises:
        ValueError: If the image_path is not valid or questions are not provided.
    """
    # Initialize the DocVQA pipeline
    doc_vqa = pipeline('document-question-answering', model='jinhybr/OCR-DocVQA-Donut')

    # Check if the image path is valid
    if not os.path.exists(image_path):
        raise ValueError(f'Image path {image_path} is not valid.')

    # Check if questions are provided
    if not questions:
        raise ValueError('No questions provided to extract information.')

    # Extract information from the insurance policy document
    answers = {}
    for question in questions:
        result = doc_vqa(image_path=image_path, question=question)
        answers[question] = result['answer']
    return answers

# test_function_code --------------------

from datasets import load_dataset


def test_extract_insurance_info():
    print("Testing started.")
    # Load a dataset with sample insurance policy document images
    dataset = load_dataset('insurance_policy_doc_images')
    sample_image_path = dataset['test'][0]['image']

    # Define questions to extract information
    questions = ['What is the policy number?', 'What is the coverage amount?', 'Who is the beneficiary?', 'What is the term period?']

    # Test if the function works
    print("Testing case [1/1] started.")
    answers = extract_insurance_info(sample_image_path, questions)
    assert answers, f"Test case failed: No answers returned."
    for question in questions:
        assert question in answers, f"Test case failed: Question '{question}' does not have an answer."
    print("Testing finished.")

# Run the test function
test_extract_insurance_info()

# call_test_function_line --------------------

test_extract_insurance_info()