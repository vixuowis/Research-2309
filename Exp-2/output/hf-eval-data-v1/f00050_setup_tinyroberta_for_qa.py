from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Function to setup a tinyroberta model from deepset for Question and Answer
# This function will return a question answering pipeline

def setup_tinyroberta_for_qa():
    model_name = 'deepset/tiny-roberta-squad2'
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    return nlp