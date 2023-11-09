from transformers import pipeline


def get_answer(question: str, context: str) -> str:
    """
    This function uses the Hugging Face Transformers library to answer questions based on a given context.

    Args:
        question (str): The question to be answered.
        context (str): The context in which the question is to be answered.

    Returns:
        str: The answer to the question.
    """
    qa_pipeline = pipeline('question-answering', model='deepset/roberta-large-squad2')
    question_context = {'question': question, 'context': context}
    answer = qa_pipeline(question_context)
    return answer['answer']