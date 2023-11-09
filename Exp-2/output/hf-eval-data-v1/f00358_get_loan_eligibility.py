from transformers import pipeline


def get_loan_eligibility(document: str, question: str) -> str:
    '''
    This function uses a pre-trained model from Hugging Face Transformers to answer questions based on a document.
    The model used is 'tiennvcs/layoutlmv2-large-uncased-finetuned-vi-infovqa', which is fine-tuned for document-based question answering tasks.
    
    Parameters:
    document (str): The document based on which the question needs to be answered.
    question (str): The question that needs to be answered.
    
    Returns:
    str: The answer to the question based on the document.
    '''
    qa_model = pipeline('question-answering', model='tiennvcs/layoutlmv2-large-uncased-finetuned-vi-infovqa')
    answer = qa_model({'question': question, 'context': document})['answer']
    return answer