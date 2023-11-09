from transformers import pipeline


def question_answering(context: str, question: str) -> str:
    """
    This function uses the 'philschmid/distilbert-onnx' model from the Hugging Face transformers library
    to answer questions based on a given context.

    Parameters:
    context (str): The context in which to search for the answer.
    question (str): The question to answer.

    Returns:
    str: The answer to the question.
    """
    # Create a question answering pipeline using the 'philschmid/distilbert-onnx' model
    qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
    # Use the pipeline to find the answer to the question in the context
    answer = qa_pipeline({'context': context, 'question': question})
    # Return the answer
    return answer['answer']