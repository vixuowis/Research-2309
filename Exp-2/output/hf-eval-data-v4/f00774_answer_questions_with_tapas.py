# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering

# function_code --------------------

def answer_questions_with_tapas(question, table):
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-base-finetuned-wikisql-supervised')
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-wikisql-supervised')
    inputs = tokenizer(table, queries=question, padding='max_length', return_tensors='pt')
    outputs = model(**inputs)
    return outputs.logits

# test_function_code --------------------

def test_answer_questions_with_tapas():
    print("Testing answer_questions_with_tapas function.")
    table = {"city": ["Amsterdam", "Berlin", "Paris"], "country": ["Netherlands", "Germany", "France"]}
    questions = ["Which country is Amsterdam in?", "What is the capital of Germany?"]

    # Call the function with the test data
    outputs = answer_questions_with_tapas(questions, table)

    # There should be 2 outputs for the 2 questions
    assert len(outputs) == 2, f"Expected 2 outputs, but got {len(outputs)}"

    print("Test passed!")

# Running the test function
test_answer_questions_with_tapas()