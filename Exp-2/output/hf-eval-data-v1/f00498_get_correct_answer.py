from transformers import pipeline

# Function to get the correct answer from multiple options using BERT model
def get_correct_answer(summary_text, question, options):
    '''
    This function takes a summary text, a question and multiple options as input.
    It uses a pre-trained BERT model to find the correct answer among the multiple options.
    The highest scoring option is returned as the correct answer.
    '''
    # Instantiate the Question Answering pipeline
    qa_pipeline = pipeline('question-answering', model='bert-large-cased-whole-word-masking-finetuned-squad')

    # Check the correct answer among the multiple options
    predictions = []
    for option in options:
        result = qa_pipeline({'context': summary_text, 'question': f'{question} {option}'})
        predictions.append((option, result['score']))

    # The highest-scoring option is the correct answer
    correct_answer = max(predictions, key=lambda x: x[1])[0]
    return correct_answer