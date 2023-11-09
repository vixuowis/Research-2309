from transformers import pipeline


def get_mars_orbit_duration(context: str, question: str) -> str:
    '''
    This function uses the 'philschmid/distilbert-onnx' model from the transformers library to answer the question about the duration of Mars' orbit around the sun.
    
    Parameters:
    context (str): The context about Mars' orbit.
    question (str): The question to be answered.
    
    Returns:
    str: The answer to the question.
    '''
    # Create a question-answering pipeline with the 'philschmid/distilbert-onnx' model
    qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
    
    # Pass the context and the question to the model
    answer = qa_pipeline({'context': context, 'question': question})
    
    # Return the answer
    return answer['answer']