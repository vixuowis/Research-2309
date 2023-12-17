# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModel, pipeline

# function_code --------------------

def answer_image_question(image_path: str, question: str) -> str:
    """
    Answers a question about an image using a pre-trained multimodal VQA model.

    Args:
        image_path (str): Path to the image.
        question (str): Question about the image.

    Returns:
        str: The answer to the question.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the question is empty or None.
    """
    model_checkpoint = 'microsoft/git-base-textvqa'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModel.from_pretrained(model_checkpoint)
    vqa_pipeline = pipeline(type='visual-question-answering', model=model, tokenizer=tokenizer)
    
    # Validate inputs
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'The image file {image_path} does not exist.')
    if not question:
        raise ValueError('The question must not be empty.')

    # Run the model and get the answer
    result = vqa_pipeline({'image': image_path, 'question': question})
    return result['answer']

# test_function_code --------------------

def test_answer_image_question():
    print("Testing started.")
    sample_image_path = 'path/to/sample_image.jpg'  # Replace with a valid image path
    sample_question = 'What is in the image?'

    # Test case 1: Valid inputs
    print("Testing case [1/3] started.")
    answer = answer_image_question(sample_image_path, sample_question)
    assert isinstance(answer, str), f"Test case [1/3] failed: Expected str, got {type(answer)}"

    # Test case 2: Image does not exist
    print("Testing case [2/3] started.")
    try:
        answer_image_question('non_existent_image.jpg', sample_question)
    except FileNotFoundError as e:
        assert 'does not exist' in str(e), f"Test case [2/3] failed: {e}"

    # Test case 3: Empty question
    print("Testing case [3/3] started.")
    try:
        answer_image_question(sample_image_path, '')
    except ValueError as e:
        assert 'The question must not be empty' in str(e), f"Test case [3/3] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_image_question()