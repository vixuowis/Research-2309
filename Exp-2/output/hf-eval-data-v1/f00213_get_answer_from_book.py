from transformers import pipeline


def get_answer_from_book(book_text: str, user_question: str) -> str:
    """
    This function uses a pre-trained model from Hugging Face to answer questions based on a given context.
    The model used is 'deepset/roberta-base-squad2-distilled', a distilled version of the Roberta model, fine-tuned for the question-answering task on the SQuAD 2.0 dataset.
    
    Parameters:
    book_text (str): The context in which to search for the answer.
    user_question (str): The question to answer.
    
    Returns:
    str: The answer to the question.
    """
    # Initialize the question answering pipeline
    qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2-distilled')
    
    # Use the pipeline to find the answer to the question in the context
    result = qa_pipeline({'context': book_text, 'question': user_question})
    
    # Extract the answer from the result
    answer = result['answer']
    
    return answer