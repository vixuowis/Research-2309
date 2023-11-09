from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


def extract_answer(question: str, context: str) -> str:
    '''
    This function uses the Hugging Face Transformers library to extract answers from a given knowledge base text.
    It uses the pre-trained DeBERTa-v3 model designed for question-answering.
    
    Args:
    question (str): The question to be answered.
    context (str): The context from which the answer will be extracted.
    
    Returns:
    str: The extracted answer.
    '''
    model_name = 'deepset/deberta-v3-large-squad2'
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    QA_input = {
        'question': question,
        'context': context
    }
    answer = nlp(QA_input)
    return answer['answer']