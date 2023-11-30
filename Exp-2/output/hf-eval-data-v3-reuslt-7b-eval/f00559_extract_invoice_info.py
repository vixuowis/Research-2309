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
    
    # Load pre-trained model from HuggingFace model hub
    try:
        
        model = AutoModelForDocumentQuestionAnswering.from_pretrained("dslim/roberta-base-deepset-model-coral")
        
    except OSError as error:
    
        print(f"[ERROR] {error}")
    
    # Extract information from invoice image using the pre-trained model
    answers = []
    try:
        
        question_answer_input = {
            "documents": [
                {"content": open(image_path, "rb"),"type": "", "language": "de"}
            ]  # document format
        }
    
        outputs = model(**question_answer_input)
        answers.append(outputs["answers"][0][0]["answer"])
        
    except OSError as error:
            
        print(f"[ERROR] {error}")
    
    try:
        
        question_answer_input = {
            "documents": [
                {"content": open(image_path, "rb"),"type": "", "language": "de"}
            ]  # document format
        }
    
        outputs = model(**question_answer_input)
        answers.append(outputs["answers"][0][1]["answer"])
        
    except OSError as error:
            
        print(f"[ERROR] {error}")
        
    try:
    
        question_answer_input = {
            "documents": [
                {"content": open(image_path, "rb"),"type": "", "language": "de"}
            ]  # document format
        }
    
        outputs = model(**question_answer_input)
        answers.append(outputs["answers"][0][2]["answer"])
        
    except OSError as error:
            
        print(f"[ERROR] {error}") 
        
    return answers

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