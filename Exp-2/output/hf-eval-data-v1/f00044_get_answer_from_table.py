from transformers import pipeline

def get_answer_from_table(question: str, table_data: dict) -> str:
    '''
    This function uses the google/tapas-small-finetuned-sqa model from the transformers library to answer questions based on tabular data.
    
    Parameters:
    question (str): The question to be answered based on the table data.
    table_data (dict): The table data in the form of a dictionary.
    
    Returns:
    str: The answer to the question based on the table data.
    '''
    # Create a table-question-answering model
    table_qa = pipeline('table-question-answering', model='google/tapas-small-finetuned-sqa')
    
    # Use the model to answer the question based on the table data
    answer = table_qa(question=question, table=table_data)
    
    return answer