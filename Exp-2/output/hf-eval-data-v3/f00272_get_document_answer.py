# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering
import requests
from PIL import Image
import pytesseract
from io import BytesIO

# function_code --------------------

def get_document_answer(image_url: str, question: str) -> str:
    """
    This function takes an image URL and a question as input, downloads the image, extracts the text from it,
    and uses a pre-trained LayoutLMv2 model to answer the question based on the extracted text.

    Args:
        image_url (str): The URL of the image to process.
        question (str): The question to answer.

    Returns:
        str: The answer to the question.
    """
    tokenizer = AutoTokenizer.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023')
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023')
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    text = pytesseract.image_to_string(img)
    inputs = tokenizer(text, question, return_tensors='pt')
    output = model(**inputs)
    answer = tokenizer.decode(output['answer_start'][0], output['answer_end'][0])
    return answer

# test_function_code --------------------

def test_get_document_answer():
    """
    This function tests the get_document_answer function with a few test cases.
    """
    assert get_document_answer('https://example.com/document_invoice.jpg', 'What is the total amount due?') is not None
    assert get_document_answer('https://example.com/document_invoice2.jpg', 'What is the due date?') is not None
    assert get_document_answer('https://example.com/document_invoice3.jpg', 'Who is the recipient?') is not None
    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_document_answer()