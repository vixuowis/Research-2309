from transformers import pipeline


def question_answering(context_text: str, question: str) -> str:
    '''
    This function uses the 'philschmid/distilbert-onnx' model from the transformers library to answer questions based on a given context.
    
    Parameters:
    context_text (str): The context in which the question should be answered.
    question (str): The question that needs to be answered.
    
    Returns:
    str: The answer to the question based on the context.
    '''
    # Load the pre-trained model fine-tuned for question answering tasks
    qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
    
    # Provide the required context and question
    answer = qa_pipeline({'context': context_text, 'question': question})
    
    return answer['answer']