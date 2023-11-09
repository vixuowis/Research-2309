from transformers import pipeline


def get_medical_answer(context: str, question: str) -> str:
    """
    This function uses the Hugging Face Transformers pipeline for question answering.
    It uses the 'sultan/BioM-ELECTRA-Large-SQuAD2' model which is specialized in biomedical language and has been fine-tuned on the SQuAD2.0 dataset.
    This makes it suitable for answering health-related questions.
    
    Parameters:
    context (str): The context in which the question is being asked.
    question (str): The question that needs to be answered.
    
    Returns:
    str: The answer to the question.
    """
    qa_pipeline = pipeline('question-answering', model='sultan/BioM-ELECTRA-Large-SQuAD2')
    result = qa_pipeline({'context': context, 'question': question})
    return result['answer']