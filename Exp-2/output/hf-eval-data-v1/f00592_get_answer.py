from transformers import pipeline

# Function to get answer to a question using BERT large cased whole word masking finetuned model on SQuAD
# @param: question - The question to be answered
# @param: context - The context in which the question is to be answered
# @return: The answer to the question

def get_answer(question: str, context: str) -> str:
    # Initialize the question-answering pipeline using the specified pretrained model 'bert-large-cased-whole-word-masking-finetuned-squad'
    qa_pipeline = pipeline('question-answering', model='bert-large-cased-whole-word-masking-finetuned-squad')
    # Pass the context and your question on price inflation to the pipeline instance which will use the pretrained model to analyze the context and generate an appropriate answer
    result = qa_pipeline({'context': context, 'question': question})
    # Return the answer
    return result['answer']