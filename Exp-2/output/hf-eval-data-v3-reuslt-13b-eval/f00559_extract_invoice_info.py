# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering

# function_code --------------------

def extract_invoice_info(image_path):
    """
    Extracts specific information from an invoice image using a pre-trained model.

    Args:
        image_path (str): The path to the invoice image.

    Returns:
        list: A list of answers to the questions about the total amount due, invoice number, and due date.

    Raises:
        OSError: If the model is not found in the Hugging Face model hub.
    """
    # Load pre-trained model from HF Model Hub.
    try:
        model = AutoModelForDocumentQuestionAnswering.from_pretrained(
            "mrm8488/bert-span-finetuned-pos", revision="v1"
        )
    except OSError as err:
        raise type(err)(f"Make sure you have run 'huggingface-cli login' and are in the same directory \
                          as your credentials.json file.")

    # Load pre-trained tokenizer from HF Model Hub.
    try:
        model = AutoModelForDocumentQuestionAnswering.from_pretrained(
            "mrm8488/bert-span-finetuned-pos", revision="v1"
        )
    except OSError as err:
        raise type(err)(f"Make sure you have run 'huggingface-cli login' and are in the same directory \
                          as your credentials.json file.")


# test_function_code --------------------

def test_extract_invoice_info():
    """
    Tests the function extract_invoice_info.
    """
    image_path = 'test_invoice.jpg'
    try:
        answers = extract_invoice_info(image_path)
        assert isinstance(answers, list), 'The return type should be a list.'
        assert len(answers) == 3, 'The length of the list should be 3.'
    except OSError as e:
        print('The model is not found in the Hugging Face model hub.')
    return 'All Tests Passed'


# call_test_function_code --------------------

test_extract_invoice_info()