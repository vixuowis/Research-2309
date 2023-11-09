from transformers import pipeline


def get_answer_from_korean_article(question: str, context: str) -> str:
    """
    This function uses a pre-trained model to answer questions based on a given context in Korean.

    Args:
        question (str): The question that needs to be answered.
        context (str): The context within which the answer is to be found.

    Returns:
        str: The answer to the question based on the context.
    """
    korean_qa = pipeline('question-answering', model='monologg/koelectra-small-v2-distilled-korquad-384')
    answer = korean_qa(question=question, context=context)
    return answer['answer']