# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering
import requests
from PIL import Image
import pytesseract
from io import BytesIO

# function_code --------------------

def get_document_answer(image_url: str, question: str) -> str:
    """
    This function uses a pre-trained LayoutLMv2 model from Hugging Face to answer questions based on a document image.

    Args:
        image_url (str): The URL of the document image.
        question (str): The question to be answered based on the document.

    Returns:
        str: The answer to the question.
    """
    tokenizer = AutoTokenizer.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023')
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023')
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    text = pytesseract.image_to_string(img)
    inputs = tokenizer(text, question, return_tensors="pt")
    output = model(**inputs)
    answer = tokenizer.decode(output["answer_start"][0], output["answer_end"][0])
    return answer

# test_function_code --------------------

def test_get_document_answer():
    """
    This function tests the get_document_answer function.
    It uses a sample image URL and a question, and checks if the returned answer is a string.
    """
    image_url = 'https://example.com/document_invoice.jpg'
    question = 'What is the total amount due?'
    answer = get_document_answer(image_url, question)
    assert isinstance(answer, str), 'The returned answer should be a string.'

# call_test_function_code --------------------

test_get_document_answer()