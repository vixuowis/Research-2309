# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# function_code --------------------

def answer_historical_question(question, context):
    """
    Answers a historical question based on the provided context using a fine-tuned Roberta model.

    Parameters:
        question (str): The question to be answered.
        context (str): The context containing information relevant to the question.

    Returns:
        str: The answer to the question.
    """
    model_name = 'deepset/roberta-base-squad2'
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    result = nlp({'question': question, 'context': context})
    return result['answer']

# test_function_code --------------------

def test_answer_historical_question():
    print("Testing the 'answer_historical_question' function.")
    question = 'What was the main cause of World War I?'
    context = 'World War I was primarily caused by a complex web of factors including ...'
    
    # Test case 1
    print("Testing case [1/1] started.")
    answer = answer_historical_question(question, context)
    assert 'assassination of Archduke Franz Ferdinand' in answer, f"Test case [1/1] failed: {answer}"
    print("Testing case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_answer_historical_question()