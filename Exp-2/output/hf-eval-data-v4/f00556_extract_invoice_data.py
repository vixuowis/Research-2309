# requirements_file --------------------

!pip install -U transformers==4.12.2 torch==1.8.0+cu101 datasets==1.14.0 tokenizers==0.10.3

# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer

# function_code --------------------

def extract_invoice_data(image_path, question):
    # Load the model
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')

    # Prepare image and question
    with open(image_path, 'rb') as image_file:
        image = image_file.read()
    
    # Process image and question with the tokenizer
    inputs = tokenizer(image, question, return_tensors='pt')
    
    # Perform inference using the model
    outputs = model(**inputs)
    
    # Extract the answer from the outputs
    answer = outputs[0]
    
    return answer

# test_function_code --------------------

def test_extract_invoice_data():
    print("Testing started.")
    # Mocked data for testing.
    test_image_path = 'invoice_sample.jpg'
    test_question = 'What is the invoice number?'

    # Test case
    print("Testing extraction of invoice number.")
    answer = extract_invoice_data(test_image_path, test_question)
    assert len(answer) > 0, "Extraction failed: No answer returned."
    print("Test for invoice number extraction passed.")

    print("Testing finished.")

# Run the test
test_extract_invoice_data()