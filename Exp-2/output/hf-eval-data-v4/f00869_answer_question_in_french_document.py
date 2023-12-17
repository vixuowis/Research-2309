# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_question_in_french_document(context, question):
    # Initialize the question-answering pipeline with a multilingual BERT model
    qa_pipeline = pipeline('question-answering', model='mrm8488/bert-multi-cased-finetuned-xquadv1', tokenizer='mrm8488/bert-multi-cased-finetuned-xquadv1')

    # Provide the text in French and the specific question to the pipeline
    answer = qa_pipeline({'context': context, 'question': question})

    # Return the answer from the pipeline
    return answer

# test_function_code --------------------

def test_answer_question_in_french_document():
    print("Testing started.")
    context = "Le nouveau bilan financier montre une augmentation des ventes de 20% au dernier trimestre."
    question = "Quelle est l'augmentation des ventes au dernier trimestre?"

    # Test case 1: Correct extraction of the sales increase percentage
    print("Testing case [1/1] started.")
    answer = answer_question_in_french_document(context, question)
    assert answer['answer'] == '20%', f"Test case [1/1] failed: Expected 20% but got {answer['answer']}"
    print("Testing finished.")

# Run the test function
test_answer_question_in_french_document()