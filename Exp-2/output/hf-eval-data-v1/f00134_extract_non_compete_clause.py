from transformers import pipeline


def extract_non_compete_clause(context: str, question: str) -> str:
    '''
    This function uses the Hugging Face Transformers library to extract information about a non-compete clause from a legal document.
    The function uses a specific CUAD-trained RoBERTa model, which is 'Rakib/roberta-base-on-cuad'.
    The context represents the legal document and the question is related to the non-compete clause.
    The model will then return the extracted answer based on the given context.
    
    Parameters:
    context (str): The legal document from which to extract the non-compete clause.
    question (str): The question related to the non-compete clause.
    
    Returns:
    str: The extracted non-compete clause.
    '''
    qa_pipeline = pipeline('question-answering', model='Rakib/roberta-base-on-cuad')
    answer = qa_pipeline(question=question, context=context)
    return answer['answer']