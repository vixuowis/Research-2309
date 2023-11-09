from transformers import pipeline, RobertaForQuestionAnswering, RobertaTokenizer


def get_covid19_answer(question: str, context: str) -> str:
    """
    This function uses a pre-trained model to answer questions related to COVID-19.

    Args:
        question (str): The question to be answered.
        context (str): The context in which the answer is to be found.

    Returns:
        str: The answer to the question.
    """
    nlp = pipeline('question-answering', model=RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2-covid'), tokenizer=RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2-covid'))
    QA_input = {'question': question, 'context': context}
    answer = nlp(QA_input)
    return answer['answer']