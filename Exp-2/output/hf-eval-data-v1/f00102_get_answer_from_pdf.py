from transformers import pipeline, LayoutLMForQuestionAnswering


def get_answer_from_pdf(image_url: str, question: str) -> str:
    """
    This function takes an image URL and a question as input, and returns the answer to the question based on the content of the image.
    The image is expected to be a screenshot of a PDF document.
    The function uses the LayoutLMForQuestionAnswering model from Hugging Face's 'impira/layoutlm-document-qa' checkpoint.
    
    Parameters:
    image_url (str): The URL of the image.
    question (str): The question to be answered.
    
    Returns:
    str: The answer to the question.
    """
    # Load the LayoutLMForQuestionAnswering model from Hugging Face's 'impira/layoutlm-document-qa' checkpoint.
    nlp = pipeline('question-answering', model=LayoutLMForQuestionAnswering.from_pretrained('impira/layoutlm-document-qa', return_dict=True))
    # Pass the image URL and the question to the pipeline to get the answer.
    result = nlp(image_url, question)
    return result