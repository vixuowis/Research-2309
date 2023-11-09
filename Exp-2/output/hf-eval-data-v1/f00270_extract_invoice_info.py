from transformers import pipeline


def extract_invoice_info(image_path):
    """
    This function uses the Hugging Face Transformers library to extract information from an invoice image.
    It uses the 'jinhybr/OCR-DocVQA-Donut' model, which is a fine-tuned model on the DocVQA dataset, designed specifically for question-answering tasks and document comprehension.
    The function takes in the path to the invoice image and returns the total amount, date of the invoice, and name of the service provider.
    
    Parameters:
    image_path (str): The path to the invoice image.
    
    Returns:
    dict: A dictionary containing the total amount, date of the invoice, and name of the service provider.
    """
    doc_vqa = pipeline('document-question-answering', model='jinhybr/OCR-DocVQA-Donut')
    questions = ['What is the total amount?', 'What is the date of the invoice?', 'What is the name of the service provider?']
    answers = [doc_vqa(image_path=image_path, question=q) for q in questions]
    return {'total_amount': answers[0], 'date_of_invoice': answers[1], 'service_provider': answers[2]}