# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# function_code --------------------

def get_answer_from_tinyroberta(question, context):
    """
    Uses the pre-trained `deepset/tiny-roberta-squad2` model from Hugging Face Transformers library
    to perform question and answering on a given context.

    :param question: The question string to be answered.
    :param context: The context string within which the answer is to be searched.
    :return: A dictionary containing the answer, score, start, and end position in the context.
    """
    # Define the model name
    model_name = 'deepset/tiny-roberta-squad2'
    # Load the question answering pipeline with the specified model
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

    # Create input dictionary
    QA_input = {'question': question, 'context': context}
    # Get the answer from the model
    result = nlp(QA_input)
    return result

# test_function_code --------------------

def test_get_answer_from_tinyroberta():
    print("Testing started.")
    # Sample input
    sample_question = 'Why is model conversion important?'
    sample_context = 'The option to convert models between FARM and transformers gives freedom to the user and lets people easily switch between frameworks.'

    # Testing case 1
    result = get_answer_from_tinyroberta(sample_question, sample_context)
    assert 'answer' in result, "Test case failed: 'answer' not in result"
    print("Test case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_get_answer_from_tinyroberta()