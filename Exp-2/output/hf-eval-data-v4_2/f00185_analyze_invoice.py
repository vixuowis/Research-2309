# requirements_file --------------------

!pip install -U transformers pillow

# function_import --------------------

from transformers import LayoutLMForQuestionAnswering, pipeline
from PIL import Image

# function_code --------------------

def analyze_invoice(image_path: str, question: str) -> dict:
    """
    Analyzes the invoice image and answers the specified question.

    Args:
        image_path (str): The path to the invoice image file.
        question (str): The question to be answered by the model.

    Returns:
        dict: The answer to the question provided by the model.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the question is empty or None.
    """
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'The image file {image_path} does not exist.')
    
    if not question:
        raise ValueError('The question must not be empty.')
    
    nlp = pipeline('question-answering', model=LayoutLMForQuestionAnswering.from_pretrained('microsoft/layoutlm-base-uncased'))
    with Image.open(image_path) as img:
        result = nlp(question, img)
        return result

# test_function_code --------------------

def test_analyze_invoice():
    from transformers import LayoutLMForQuestionAnswering, pipeline
    
    print("Testing started.")
    
    # Test case 1: Correct Invoice Image and Question
    print("Testing case [1/3] started.")
    result = analyze_invoice('valid_invoice_image.jpg', 'What is the total amount?')
    assert 'answer' in result, f"Test case [1/3] failed: The key 'answer' should be in the result."

    # Test case 2: Non-existing Invoice Image
    print("Testing case [2/3] started.")
    try:
        analyze_invoice('non_existing_image.jpg', 'What is the due date?')
        assert False, "Test case [2/3] failed: FileNotFoundError should have been raised."
    except FileNotFoundError:
        pass

    # Test case 3: Empty Question
    print("Testing case [3/3] started.")
    try:
        analyze_invoice('valid_invoice_image.jpg', '')
        assert False, "Test case [3/3] failed: ValueError should have been raised."
    except ValueError:
        pass
    
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_invoice()