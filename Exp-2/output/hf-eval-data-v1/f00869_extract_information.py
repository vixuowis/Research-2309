from transformers import pipeline


def extract_information(context: str, question: str) -> dict:
    """
    Extracts specific information from a French business document using a multilingual question-answering model.

    Args:
        context (str): The French text document from which to extract information.
        question (str): The specific question in French to answer based on the text document.

    Returns:
        dict: The answer to the question based on the information provided in the text document.
    """
    qa_pipeline = pipeline('question-answering', model='mrm8488/bert-multi-cased-finetuned-xquadv1', tokenizer='mrm8488/bert-multi-cased-finetuned-xquadv1')
    answer = qa_pipeline({'context': context, 'question': question})
    return answer