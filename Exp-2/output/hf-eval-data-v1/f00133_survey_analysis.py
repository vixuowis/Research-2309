from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer, pipeline

# Load model & tokenizer
tapas_model = AutoModelForTableQuestionAnswering.from_pretrained('navteca/tapas-large-finetuned-wtq')
tapas_tokenizer = AutoTokenizer.from_pretrained('navteca/tapas-large-finetuned-wtq')

# Get predictions
nlp = pipeline('table-question-answering', model=tapas_model, tokenizer=tapas_tokenizer)

# Function to analyze survey results
def survey_analysis(survey_table, query):
    """This function takes a survey table and a query as input and returns the number of respondents who have given a specific answer for each question.
    Args:
        survey_table (dict): The survey table in appropriate format.
        query (str): The query to be asked.
    Returns:
        dict: The result of the query.
    """
    result = nlp({'table': survey_table, 'query': query})
    return result