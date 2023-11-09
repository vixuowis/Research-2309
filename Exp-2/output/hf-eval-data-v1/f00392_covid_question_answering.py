from transformers import pipeline, RobertaForQuestionAnswering, RobertaTokenizer


def covid_question_answering(question: str, context: str) -> str:
    """
    This function uses a pre-trained Roberta model to answer questions about COVID-19.
    The model has been fine-tuned specifically for question answering tasks about the COVID-19 pandemic and its related research papers.
    
    Args:
        question (str): The question to be answered.
        context (str): The context in which the question is to be answered.
    
    Returns:
        str: The answer to the question.
    """
    # Load the pre-trained Roberta model and tokenizer
    qa_pipeline = pipeline(
        'question-answering', 
        model=RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2-covid'), 
        tokenizer=RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2-covid')
    )
    
    # Use the model to answer the question
    answer = qa_pipeline({'question': question, 'context': context})
    
    return answer['answer']