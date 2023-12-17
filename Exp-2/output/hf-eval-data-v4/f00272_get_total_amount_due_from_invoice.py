# requirements_file --------------------

!pip install -U transformers>=4.11.0 requests Pillow pytesseract

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering
import requests
from PIL import Image
from io import BytesIO
import pytesseract

# function_code --------------------

tokenizer = AutoTokenizer.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023')
model = AutoModelForDocumentQuestionAnswering.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023')

def get_total_amount_due_from_invoice(image_url):
    """
    Given the URL of a document image, this function asks the question "What is the total amount due?" 
    and uses a pre-trained model to extract the answer from the image.

    Params:
    - image_url (str): The URL of the document image.

    Returns:
    - str: The answer extracted from the image.
    """

    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    text = pytesseract.image_to_string(img)
    inputs = tokenizer(text, "What is the total amount due?", return_tensors="pt")
    outputs = model(**inputs)
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
    answer_start = answer_start_scores.argmax()
    answer_end = answer_end_scores.argmax() + 1
    answer = tokenizer.decode(inputs.input_ids[0][answer_start:answer_end])

    return answer

# test_function_code --------------------

def test_get_total_amount_due_from_invoice():
    print("Testing started.")
    sample_image_url = "https://example.com/sample_invoice.jpg"
    print("Testing case [1/1] started.")
    total_amount_due = get_total_amount_due_from_invoice(sample_image_url)
    assert total_amount_due != '', f"Test case [1/1] failed: Expected non-empty string, got '{total_amount_due}'"
    print("Testing finished.")

test_get_total_amount_due_from_invoice()