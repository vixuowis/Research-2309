from transformers import AutoModelForDocumentQuestionAnswering


def extract_invoice_info(image):
    '''
    This function uses a pre-trained model from Hugging Face Transformers to extract specific information from an invoice image.
    The information includes total amount due, invoice number, and due date.
    
    Args:
    image (str): The path to the invoice image.
    
    Returns:
    list: A list of answers to the questions.
    '''
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023')
    inputs, layout = preprocess_image(image) # a custom function to preprocess the image 
    questions = ['What is the total amount due?', 'What is the invoice number?', 'What is the due date?']
    answers = []
    for question in questions:
        answer = model(inputs, layout, question)
        answers.append(answer)
    return answers