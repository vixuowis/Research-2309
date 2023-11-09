from transformers import pipeline


def get_answer(question: str, context: str) -> str:
    '''
    This function uses the Hugging Face Transformers library to answer questions based on a given context.
    It uses the 'bert-large-uncased-whole-word-masking-finetuned-squad' model which is trained to answer questions based on context.
    
    Args:
    question (str): The question to be answered.
    context (str): The context in which to find the answer.
    
    Returns:
    str: The answer to the question.
    '''
    # Create a question-answering model
    qa_model = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')
    
    # Pass a dictionary containing the question and context as input to the model
    result = qa_model({'question': question, 'context': context})
    
    # The model will find the most likely answer in the provided context
    answer = result['answer']
    
    return answer