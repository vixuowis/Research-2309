# requirements_file --------------------

!pip install -U transformers requests Pillow pytesseract

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering
import requests
from PIL import Image
import pytesseract
from io import BytesIO

# function_code --------------------

def extract_total_amount_due(image_url, question):
    """
    Extracts the total amount due from a document image using a pre-trained LayoutLMX model.

    Args:
        image_url (str): The URL of the document image.
        question (str): The question to be asked regarding the total amount due.

    Returns:
        str: The extracted answer from the document image.

    Raises:
        ValueError: If the image cannot be retrieved.
    """
    tokenizer = AutoTokenizer.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023')
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023')

    # Attempt to get the image from the URL
    response = requests.get(image_url)
    if response.status_code != 200:
        raise ValueError('Image cannot be retrieved.')

    # Read the image
    img = Image.open(BytesIO(response.content))

    # Use pytesseract to extract text and layout information
    text = pytesseract.image_to_string(img)

    # Tokenize the text and the question
    inputs = tokenizer(text, question, return_tensors='pt')

    # Get the model's answer
    outputs = model(**inputs)
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits

    # Find the tokens with the highest start and end scores
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # Convert tokens to the answer string
    answer = tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end])

    return answer

# test_function_code --------------------

def test_extract_total_amount_due():
    print('Testing started.')
    # Test case 1
    print('Testing case [1/1] started.')
    example_image_url = 'https://example.com/document_invoice.jpg'
    example_question = 'What is the total amount due?'
    try:
        answer = extract_total_amount_due(example_image_url, example_question)
        assert isinstance(answer, str), f'Test case [1/1] failed: The function should return a string, but got {type(answer)}.'
    except Exception as e:
        assert False, f'Test case [1/1] failed with exception: {e}'
    print('Testing finished.')

# call_test_function_line --------------------

test_extract_total_amount_due()