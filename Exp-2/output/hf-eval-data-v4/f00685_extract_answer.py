# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# function_code --------------------

def extract_answer(question, context):
    # Load the pre-trained DeBERTa-v3 model for question answering
    model_name = 'deepset/deberta-v3-large-squad2'
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create a question answering pipeline
    qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

    # Prepare the input
    QA_input = {
        'question': question,
        'context': context
    }

    # Get the answer from the model
    answer = qa_pipeline(QA_input)
    return answer

# test_function_code --------------------

def test_extract_answer():
    print("Testing started.")
    # Test case: 'What are the benefits of exercise?'
    question = 'What are the benefits of exercise?'
    context = 'Exercise helps maintain a healthy body weight, improves cardiovascular health, and boosts the immune system.'
    expected_answer = 'maintain a healthy body weight, improves cardiovascular health, and boosts the immune system'

    print("Testing case [1/1] started.")
    result = extract_answer(question, context)
    assert expected_answer in result['answer'], f"Test case failed: Expected answer fragment '{expected_answer}' not found in the result {result['answer']}"
    print("Test case [1/1] passed.")
    print("Testing finished.")

test_extract_answer()