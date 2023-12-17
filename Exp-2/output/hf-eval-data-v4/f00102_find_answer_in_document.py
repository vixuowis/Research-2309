# requirements_file --------------------

!pip install -U transformers requests Pillow pytesseract

# function_import --------------------

from transformers import pipeline, LayoutLMForQuestionAnswering
from PIL import Image
import pytesseract
import requests
from io import BytesIO

# function_code --------------------

def find_answer_in_document(image_url, question):
    """
    Find the answer to a question in a document represented by an image URL.

    Parameters:
    image_url (str): The URL of the image representing the document.
    question (str): The question to find an answer for in the document.

    Returns:
    dict: The result containing the answer.
    """
    # Load the question-answering pipeline with the LayoutLM model
    nlp = pipeline('question-answering', model=LayoutLMForQuestionAnswering.from_pretrained('impira/layoutlm-document-qa', return_dict=True))

    # Open the image from the URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image = image.convert('RGB')  # Ensure image is in RGB

    # Use pytesseract to perform OCR if needed
    # text = pytesseract.image_to_string(image)

    # Get the answer from the question
    result = nlp(question=question, image=image)
    return result

# test_function_code --------------------

def test_find_answer_in_document():
    print("Testing started.")

    # Define image URL and question for testing
    image_url = 'https://templates.invoicehome.com/invoice-template-us-neat-750px.png'
    question = 'What is the invoice number?'

    # Expected output format: {'score': float, 'start': int, 'end': int, 'answer': str}
    # Expected answer should be known for a valid test, assuming a specific document
    expected_answer = '1234567890'

    # Testing case
    print("Testing case started.")
    result = find_answer_in_document(image_url, question)
    assert result['answer'] == expected_answer, f"Test case failed: Expected answer was '{expected_answer}', but got '{result['answer']}'"
    print("Testing finished.")

# Run the test function
test_find_answer_in_document()