from transformers import pipeline


def get_answer_from_text(question: str, context: str) -> str:
    """
    This function uses a pre-trained model from the transformers library to answer questions based on a given context.

    Args:
        question (str): The question to be answered.
        context (str): The text from which the answer will be extracted.

    Returns:
        str: The extracted answer.
    """
    question_answerer = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
    result = question_answerer(question=question, context=context)
    return result['answer']