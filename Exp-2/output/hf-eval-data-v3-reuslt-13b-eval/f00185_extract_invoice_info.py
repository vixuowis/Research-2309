# function_import --------------------

from transformers import LayoutLMForQuestionAnswering, pipeline
from PIL import Image

# function_code --------------------

def extract_invoice_info(image_path: str, question: str) -> str:
    """
    Extracts information from an invoice image using a question-answering model.

    Args:
        image_path (str): The path to the invoice image.
        question (str): The question to be answered based on the invoice image.

    Returns:
        str: The answer to the question based on the invoice image.
    """
    nlp = pipeline(task="question-answering", model='microsoft/layoutlm-base-uncased-finetuned-posnegclassification')

    with open(image_path, "rb") as f:
        byte_im = f.read()
    pil_im = Image.open(io.BytesIO(byte_im))
    result = nlp({'question': question,'context':pil_im})
    return result['answer']
# function_code --------------------

# test_function_code --------------------

def test_extract_invoice_info():
    """
    Tests the function 'extract_invoice_info'.
    """
    # Test case 1
    image_path = 'https://templates.invoicehome.com/invoice-template-us-neat-750px.png'
    question = 'What is the invoice number?'
    assert isinstance(extract_invoice_info(image_path, question), str)

    # Test case 2
    image_path = 'https://miro.medium.com/max/787/1*iECQRIiOGTmEFLdWkVIH2g.jpeg'
    question = 'What is the purchase amount?'
    assert isinstance(extract_invoice_info(image_path, question), str)

    # Test case 3
    image_path = 'https://www.accountingcoach.com/wp-content/uploads/2013/10/income-statement-example@2x.png'
    question = 'What are the 2020 net sales?'
    assert isinstance(extract_invoice_info(image_path, question), str)

    return 'All Tests Passed'


# call_test_function_code --------------------

test_extract_invoice_info()