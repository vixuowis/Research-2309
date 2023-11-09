from transformers import pipeline


def get_answer_from_textbook(question: str, textbook_content: str) -> str:
    '''
    This function uses the 'distilbert-base-uncased-distilled-squad' model from the transformers library
    to answer questions based on the provided textbook content.
    
    Args:
    question (str): The question to be answered.
    textbook_content (str): The textbook content to find the answer from.
    
    Returns:
    str: The answer to the question.
    '''
    # Create a question-answering model
    qa_model = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
    
    # Use the model to find the answer to the question within the textbook content
    result = qa_model(question=question, context=textbook_content)
    
    # Extract the answer from the result
    answer = result['answer']
    
    return answer