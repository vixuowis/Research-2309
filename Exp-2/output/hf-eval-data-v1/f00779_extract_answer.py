from transformers import pipeline


def extract_answer(question: str, context: str) -> str:
    """
    This function uses a pre-trained model from Hugging Face Transformers to extract answers from a given context based on a question.

    Args:
        question (str): The question for which an answer is needed.
        context (str): The context from which the answer should be extracted.

    Returns:
        str: The extracted answer.
    """
    nlp = pipeline('question-answering', model='deepset/deberta-v3-large-squad2', tokenizer='deepset/deberta-v3-large-squad2')
    QA_input = {'question': question, 'context': context}
    result = nlp(QA_input)
    return result['answer']