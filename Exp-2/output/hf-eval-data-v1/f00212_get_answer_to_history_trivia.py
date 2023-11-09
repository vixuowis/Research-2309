from transformers import pipeline


def get_answer_to_history_trivia(context: str, question: str) -> str:
    """
    This function uses a question answering model to provide answers to trivia questions about history.
    The model has been fine-tuned on the question-answering task using a version of the DistilBERT-base-cased model.
    
    Parameters:
    context (str): The context in which the answer to the question is contained.
    question (str): The trivia question.
    
    Returns:
    str: The answer to the trivia question.
    """
    # Create a question answering pipeline with the specified model
    qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
    
    # Use the pipeline to get the answer to the trivia question
    answer = qa_pipeline({'context': context, 'question': question})
    
    return answer['answer']