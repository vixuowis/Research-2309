from transformers import TapasForQuestionAnswering


def get_company_revenue(question, table_data):
    """
    This function uses the TAPAS model to answer questions based on tabular data.
    The model has been fine-tuned for answering questions based on the WikiSQL dataset.
    
    Args:
    question (str): The question to be answered.
    table_data (list): The table data in the form of a list of dictionaries.
    
    Returns:
    str: The answer to the question.
    """
    # Load the TAPAS model
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-wikisql-supervised')
    
    # Predict the answer
    answer = model.predict(question, table_data)
    
    return answer