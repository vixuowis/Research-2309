from transformers import pipeline, AutoModel, AutoTokenizer

# This function is designed for extractive question answering and supports English language.
# It uses the bert-large model, fine-tuned using the SQuAD2.0 dataset.
# The model and tokenizer are loaded from the 'deepset/bert-large-uncased-whole-word-masking-squad2' pretrained model.
def question_answering(question, context):
    # Initialize the question answering pipeline
    nlp = pipeline('question-answering', model=AutoModel.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2'), tokenizer=AutoTokenizer.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2'))
    # Prepare the input data
    QA_input = {
        'question': question,
        'context': context
    }
    # Get the answer from the model
    res = nlp(QA_input)
    return res