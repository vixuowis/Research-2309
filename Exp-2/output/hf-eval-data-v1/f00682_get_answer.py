from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Function to get answer for a given historical question
# This function uses the 'deepset/roberta-base-squad2' model from Hugging Face Transformers
# The model is fine-tuned on the SQuAD 2.0 dataset for question answering tasks
# The function takes a question and a context as input and returns the answer

def get_answer(question, context):
    model_name = 'deepset/roberta-base-squad2'
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
     'question': question,
     'context': context
    }
    res = nlp(QA_input)
    return res['answer']